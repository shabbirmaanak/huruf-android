// Bridge between Java (HurufSpeech) and whisper.cpp.
#include <jni.h>
#include <string>
#include <thread>
#include <algorithm>
#include <android/log.h>
#include "whisper.h"

#define LOG(...) __android_log_print(ANDROID_LOG_DEBUG, "HurufWhisper", __VA_ARGS__)

extern "C" JNIEXPORT jlong JNICALL
Java_com_shabbirmaanak_huruf_HurufSpeech_nInit(JNIEnv* env, jclass, jstring jpath) {
    const char* path = env->GetStringUTFChars(jpath, nullptr);
    whisper_context_params cp = whisper_context_default_params();
    cp.use_gpu = false;
    whisper_context* ctx = whisper_init_from_file_with_params(path, cp);
    env->ReleaseStringUTFChars(jpath, path);
    LOG("model %s", ctx ? "loaded" : "FAILED");
    return (jlong) ctx;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_shabbirmaanak_huruf_HurufSpeech_nRun(JNIEnv* env, jclass, jlong h, jfloatArray jpcm, jint threads) {
    auto* ctx = (whisper_context*) h;
    if (!ctx) return env->NewStringUTF("");
    jsize n = env->GetArrayLength(jpcm);
    jfloat* pcm = env->GetFloatArrayElements(jpcm, nullptr);
    whisper_full_params p = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    p.language = "ar";
    p.translate = false;
    p.n_threads = threads;
    p.no_context = true;
    p.no_timestamps = true;
    p.single_segment = true;
    p.print_progress = false;
    p.print_realtime = false;
    p.print_special = false;
    p.print_timestamps = false;
    p.suppress_blank = true;
    // pieces are at most ~9.5 s, so a shorter audio window is enough and much faster
    p.audio_ctx = n <= 16000 * 10 ? 512 : 0;
    std::string out;
    if (whisper_full(ctx, p, pcm, n) == 0) {
        int segs = whisper_full_n_segments(ctx);
        for (int i = 0; i < segs; i++) out += whisper_full_get_segment_text(ctx, i);
    }
    env->ReleaseFloatArrayElements(jpcm, pcm, JNI_ABORT);
    return env->NewStringUTF(out.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_shabbirmaanak_huruf_HurufSpeech_nFree(JNIEnv*, jclass, jlong h) {
    if (h) whisper_free((whisper_context*) h);
}
