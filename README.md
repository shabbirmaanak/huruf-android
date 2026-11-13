# Huruf Android app

Wraps `../index.html` (repo `shabbirmaanak/huruf-server`) with Capacitor 7.

- Update the app: `npm run sync`, then build in `android/` with `./gradlew assembleDebug` (JAVA_HOME = Android Studio's JBR).
- Tasmee' listening: `HurufSpeech` plugin (android/app/src/main/java/.../HurufSpeech.java) runs
  Tarteel's Quran model with whisper.cpp v1.7.6 (MIT, copied unchanged in android/app/src/main/cpp/whisper.cpp).

## Rebuilding the model file (not stored in git, ~60 MB)

1. Download tarteel-ai/whisper-base-ar-quran (Apache 2.0) from Hugging Face.
2. `python -I tools/convert_safe.py <model_dir> <openai-whisper>/whisper/assets/mel_filters.npz ggml-quran-base-f16.bin`
3. Build whisper.cpp's `quantize` and run `quantize ggml-quran-base-f16.bin quran-base-q5_1.bin q5_1`.
4. Put `quran-base-q5_1.bin` in `android/app/src/main/assets/`.
