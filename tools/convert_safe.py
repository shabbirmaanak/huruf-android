# Converts tarteel-ai/whisper-base-ar-quran (Hugging Face format) to whisper.cpp's ggml file.
# Same file layout as whisper.cpp/models/convert-h5-to-ggml.py, but the weights are opened with
# torch.load(weights_only=True), so nothing inside the downloaded file can run as code.
# Usage: python -I convert_safe.py <model_dir> <mel_filters.npz> <out_file>
import json, struct, sys
from pathlib import Path
import numpy as np
import torch

dir_model, mel_path, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])

conv_map = {
    'self_attn.k_proj': 'attn.key', 'self_attn.q_proj': 'attn.query', 'self_attn.v_proj': 'attn.value',
    'self_attn.out_proj': 'attn.out', 'self_attn_layer_norm': 'attn_ln',
    'encoder_attn.q_proj': 'cross_attn.query', 'encoder_attn.v_proj': 'cross_attn.value',
    'encoder_attn.out_proj': 'cross_attn.out', 'encoder_attn_layer_norm': 'cross_attn_ln',
    'fc1': 'mlp.0', 'fc2': 'mlp.2', 'final_layer_norm': 'mlp_ln',
    'encoder.layer_norm.bias': 'encoder.ln_post.bias', 'encoder.layer_norm.weight': 'encoder.ln_post.weight',
    'encoder.embed_positions.weight': 'encoder.positional_embedding',
    'decoder.layer_norm.bias': 'decoder.ln.bias', 'decoder.layer_norm.weight': 'decoder.ln.weight',
    'decoder.embed_positions.weight': 'decoder.positional_embedding',
    'decoder.embed_tokens.weight': 'decoder.token_embedding.weight',
}

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

hp = json.load(open(dir_model / "config.json", encoding="utf8"))
max_length = hp.get("max_target_positions", 448)
tokens = json.load(open(dir_model / "vocab.json", encoding="utf8"))
with np.load(mel_path) as f:   # allow_pickle is False by default
    filters = f[f"mel_{hp['num_mel_bins']}"]

state = torch.load(dir_model / "pytorch_model.bin", map_location="cpu", weights_only=True)

with open(out_path, "wb") as fout:
    fout.write(struct.pack("i", 0x67676d6c))
    for v in [hp["vocab_size"], hp["max_source_positions"], hp["d_model"], hp["encoder_attention_heads"],
              hp["encoder_layers"], max_length, hp["d_model"], hp["decoder_attention_heads"],
              hp["decoder_layers"], hp["num_mel_bins"], 1]:
        fout.write(struct.pack("i", int(v)))
    fout.write(struct.pack("ii", filters.shape[0], filters.shape[1]))
    fout.write(filters.astype(np.float32).tobytes())

    dec = {v: k for k, v in bytes_to_unicode().items()}
    fout.write(struct.pack("i", len(tokens)))
    for key, _ in sorted(tokens.items(), key=lambda x: x[1]):
        text = bytes([dec[c] for c in key])
        fout.write(struct.pack("i", len(text))); fout.write(text)

    count = 0
    for src, t in state.items():
        if src == "proj_out.weight":
            continue
        nn = src.split(".")[1:]   # drop leading "model."
        if nn[1] == "layers":
            nn[1] = "blocks"
            sub = ".".join(nn[3:-1])
            mapped = ("attn.key" if nn[0] == "encoder" else "cross_attn.key") if sub == "encoder_attn.k_proj" else conv_map[sub]
            name = ".".join(nn[:3] + [mapped] + nn[-1:])
        else:
            name = ".".join(nn); name = conv_map.get(name, name)
        data = t.squeeze().float().numpy().astype(np.float16)
        if name in ("encoder.conv1.bias", "encoder.conv2.bias"):
            data = data.reshape(data.shape[0], 1)
        n_dims = len(data.shape); ftype = 1
        if n_dims < 2 or name in ("encoder.conv1.bias", "encoder.conv2.bias", "encoder.positional_embedding", "decoder.positional_embedding"):
            data = data.astype(np.float32); ftype = 0
        b = name.encode("utf-8")
        fout.write(struct.pack("iii", n_dims, len(b), ftype))
        for i in range(n_dims):
            fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
        fout.write(b); data.tofile(fout); count += 1
print("tensors written:", count, "->", out_path)
