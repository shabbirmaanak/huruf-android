package com.shabbirmaanak.huruf;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.res.AssetFileDescriptor;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.util.Log;
import com.getcapacitor.JSObject;
import com.getcapacitor.PermissionState;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;
import com.getcapacitor.annotation.Permission;
import com.getcapacitor.annotation.PermissionCallback;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

/*
 * Tasmee' listening, fully on the phone.
 * The microphone audio is cut into pieces at natural pauses (or every ~9 s) and each piece is written out by
 * Tarteel's Quran recitation model (tarteel-ai/whisper-base-ar-quran, Apache 2.0) running in whisper.cpp (MIT).
 * Nothing is sent over the internet.
 * Events to the page: "partial" {text} = rough reading of the piece still being recited (repeated as it grows),
 * "final" {text} = finished reading of that piece, "state" {state:"idle", reason} = listening ended.
 */
@CapacitorPlugin(
    name = "HurufSpeech",
    permissions = { @Permission(strings = { Manifest.permission.RECORD_AUDIO }, alias = "mic") }
)
public class HurufSpeech extends Plugin {

    static { System.loadLibrary("hurufwhisper"); }
    private static native long nInit(String modelPath);
    private static native String nRun(long ctx, float[] pcm, int threads);
    private static native void nFree(long ctx);

    private static final String TAG = "HurufSpeech";
    private static final String MODEL = "quran-base-q5_1.bin";
    private static final int RATE = 16000, FRAME = 1600;          // 100 ms frames
    private static final int END_SILENCE_FRAMES = 6;               // 0.6 s quiet ends a piece
    private static final int MAX_PIECE = RATE * 9;                 // cut long pieces at 9 s
    private static final int PARTIAL_EVERY = RATE * 1;             // rough reading about every second

    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private final AtomicInteger queued = new AtomicInteger();
    private volatile long ctx = 0;
    private volatile boolean want = false;
    private Thread recThread;
    private final int threads = Math.max(2, Math.min(4, Runtime.getRuntime().availableProcessors() - 2));

    @PluginMethod
    public void check(PluginCall call) {
        JSObject r = new JSObject();
        r.put("available", true);
        r.put("mic", getPermissionState("mic") == PermissionState.GRANTED ? "granted" : "prompt");
        call.resolve(r);
    }

    @PluginMethod
    public void askMic(PluginCall call) {
        if (getPermissionState("mic") == PermissionState.GRANTED) call.resolve(new JSObject().put("mic", "granted"));
        else requestPermissionForAlias("mic", call, "micDone");
    }

    @PermissionCallback
    private void micDone(PluginCall call) {
        call.resolve(new JSObject().put("mic", getPermissionState("mic") == PermissionState.GRANTED ? "granted" : "denied"));
    }

    @PluginMethod
    public void start(PluginCall call) {
        if (getPermissionState("mic") != PermissionState.GRANTED) { call.reject("not granted", "not_granted"); return; }
        if (want) { call.resolve(); return; }
        want = true;
        worker.execute(() -> {   // load the model once (copied out of the app the first time)
            if (ctx == 0) {
                try {
                    File f = modelFile();
                    long t0 = System.currentTimeMillis();
                    ctx = nInit(f.getAbsolutePath());
                    Log.d(TAG, "model load " + (System.currentTimeMillis() - t0) + " ms");
                } catch (Exception e) { Log.e(TAG, "model copy failed", e); }
            }
            if (ctx == 0) { want = false; idle("unavailable"); }
        });
        recThread = new Thread(this::recordLoop, "huruf-mic");
        recThread.start();
        call.resolve();
    }

    @PluginMethod
    public void stop(PluginCall call) {
        want = false;
        call.resolve();
    }

    @Override
    protected void handleOnPause() {
        super.handleOnPause();
        if (want) { want = false; idle("hidden"); }
    }

    @Override
    protected void handleOnDestroy() {
        want = false;
        worker.execute(() -> { if (ctx != 0) { nFree(ctx); ctx = 0; } });
        worker.shutdown();
        super.handleOnDestroy();
    }

    private void idle(String reason) {
        JSObject o = new JSObject().put("state", "idle");
        if (reason != null) o.put("reason", reason);
        notifyListeners("state", o);
    }

    private File modelFile() throws Exception {
        File out = new File(getContext().getFilesDir(), MODEL);
        long size;
        try (AssetFileDescriptor fd = getContext().getAssets().openFd(MODEL)) { size = fd.getLength(); }
        if (out.exists() && out.length() == size) return out;
        File tmp = new File(out.getPath() + ".part");
        try (InputStream in = getContext().getAssets().open(MODEL); OutputStream os = new FileOutputStream(tmp)) {
            byte[] buf = new byte[1 << 16]; int n;
            while ((n = in.read(buf)) > 0) os.write(buf, 0, n);
        }
        if (!tmp.renameTo(out)) throw new Exception("rename failed");
        return out;
    }

    /* ---------- microphone: cut the recitation into pieces ---------- */
    @SuppressLint("MissingPermission")
    private void recordLoop() {
        int min = AudioRecord.getMinBufferSize(RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
        AudioRecord ar;
        try {
            ar = new AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION, RATE, AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT, Math.max(min, FRAME * 2 * 4));
            ar.startRecording();
        } catch (Exception e) {
            Log.e(TAG, "mic failed", e);
            want = false; idle("upstream_error"); return;
        }
        short[] frame = new short[FRAME];
        float[] piece = new float[MAX_PIECE + FRAME];
        float[][] pre = new float[3][FRAME]; int preN = 0;   // 0.3 s kept from before speech starts
        int len = 0, quiet = 0, sinceRough = 0;
        boolean speaking = false;
        float floor = 0.004f;
        while (want) {
            int got = 0;
            while (got < FRAME && want) { int r = ar.read(frame, got, FRAME - got); if (r < 0) break; got += r; }
            if (got < FRAME) break;
            float[] f = new float[FRAME]; double sum = 0;
            for (int i = 0; i < FRAME; i++) { f[i] = frame[i] / 32768f; sum += f[i] * f[i]; }
            float rms = (float) Math.sqrt(sum / FRAME);
            float thresh = Math.max(0.008f, floor * 2.5f);
            boolean loud = rms > thresh;
            if (!speaking) {
                floor = floor * 0.95f + Math.min(rms, 0.05f) * 0.05f;
                if (loud) {
                    speaking = true; len = 0; quiet = 0; sinceRough = 0;
                    for (int k = Math.max(0, preN - 3); k < preN; k++) { System.arraycopy(pre[k % 3], 0, piece, len, FRAME); len += FRAME; }
                } else { pre[preN % 3] = f; preN++; continue; }
            }
            System.arraycopy(f, 0, piece, len, FRAME); len += FRAME; sinceRough += FRAME;
            quiet = loud ? 0 : quiet + 1;
            if (quiet >= END_SILENCE_FRAMES) {          // natural pause: finish this piece
                send(piece, len - (quiet - 2) * FRAME, true);
                speaking = false; preN = 0;
            } else if (len >= MAX_PIECE) {               // long breath: cut and carry straight on
                send(piece, len, true);
                len = 0; sinceRough = 0;
            } else if (sinceRough >= PARTIAL_EVERY && queued.get() == 0) {
                send(piece, len, false);
                sinceRough = 0;
            }
        }
        try { ar.stop(); ar.release(); } catch (Exception e) {}
        if (speaking && len > RATE / 2) send(piece, len, true);
        worker.execute(() -> idle(null));
    }

    private void send(float[] piece, int len, boolean fin) {
        if (len < RATE * 2 / 5) return;                  // shorter than 0.4 s: a click or breath
        final float[] pcm = new float[len];
        System.arraycopy(piece, 0, pcm, 0, len);
        queued.incrementAndGet();
        worker.execute(() -> {
            try {
                if (ctx == 0) return;
                long t0 = System.currentTimeMillis();
                String text = nRun(ctx, pcm, threads).trim();
                Log.d(TAG, (fin ? "final " : "rough ") + String.format("%.1f", len / (float) RATE) + "s in "
                    + (System.currentTimeMillis() - t0) + "ms: " + text);
                if (!text.isEmpty()) notifyListeners(fin ? "final" : "partial", new JSObject().put("text", text));
            } finally { queued.decrementAndGet(); }
        });
    }
}
