"""Capacity math: store state vs text bytes vs KV cache, for the 4B config (config.json)."""
import json
import os

from common import (Config, CMAX, ID_BITS, ROOT, MODEL_DIR, load_tokenizer, encode, write_section)

tok = load_tokenizer()
cfg_json = json.load(open(os.path.join(MODEL_DIR, "config.json"), encoding="utf-8"))
tc = cfg_json.get("text_config", cfg_json)
layers = tc["layer_types"]
n_full = sum(1 for t in layers if t == "full_attention")
n_lin = sum(1 for t in layers if t == "linear_attention")
kv_heads, head_dim = tc["num_key_value_heads"], tc["head_dim"]
BYTES = 2  # bf16 KV
kv_per_token = n_full * 2 * kv_heads * head_dim * BYTES

N_FILES = 512
TOK = N_FILES * CMAX  # 53248
cfg = Config()

# store state (logical bits of the reference dicts in store.py Device)
name_key_bits = 6 * ID_BITS + 3
slot_bits = name_key_bits + 1 + 9 + 9  # key + alive + head + tail (512 chunks -> 9 bits)
dir_bits = cfg.n_slots * slot_bits
order_bits = cfg.n_names * 10  # slot index (1024 slots)
len_bits = cfg.n_chunks * 7  # text length 0..104
next_bits = cfg.n_chunks * 9
content_bits = N_FILES * CMAX * ID_BITS  # 18 bits per stored token id
store_bits = content_bits + dir_bits + order_bits + len_bits + next_bits
store_bytes = store_bits / 8

# text bytes: utf-8 bytes per Qwen token, measured on docs/research/parametric-memory.md (read-only)
txt = open(os.path.join(ROOT, "docs", "research", "parametric-memory.md"), encoding="utf-8").read()
ids = encode(tok, txt)
bpt = len(txt.encode("utf-8")) / len(ids)
text_bytes = TOK * bpt

# GDN fixed recurrent state of the model (not per-file; the store uses a fixed subset of heads)
gdn_state_fp32 = n_lin * tc["linear_num_value_heads"] * tc["linear_value_head_dim"] * tc["linear_key_head_dim"] * 4

out = {
    "config_source": "C:/Projects/Qcompute/.data/Qwen3.5-4B-Qwenputer-Filesystem/config.json",
    "layers": {"total": len(layers), "full_attention": n_full, "linear_attention": n_lin},
    "kv_heads": kv_heads, "head_dim": head_dim, "kv_dtype_bytes": BYTES,
    "kv_bytes_per_token": kv_per_token,
    "docs_assumption_kv_bytes_per_token": 32768,
    "docs_assumption_note": "docs/research/parametric-memory.md assumes 8 full-attention layers (32 KB/token); config has %d" % n_full,
    "files": N_FILES, "tokens_per_file_cmax": CMAX, "total_tokens": TOK,
    "store_logical_bits": {"content": content_bits, "directory": dir_bits, "order": order_bits,
                           "lengths": len_bits, "next_pointers": next_bits, "total": store_bits},
    "store_logical_bytes": store_bytes,
    "store_logical_kib": round(store_bytes / 1024, 2),
    "text_utf8_bytes_per_token_measured": round(bpt, 4),
    "text_bytes_for_512x104_tokens": round(text_bytes),
    "text_kib": round(text_bytes / 1024, 1),
    "kv_cache_bytes_for_53248_tokens": TOK * kv_per_token,
    "kv_cache_gib_for_53248_tokens": round(TOK * kv_per_token / 2**30, 3),
    "gdn_fixed_recurrent_state_bytes_fp32_all_linear_layers": gdn_state_fp32,
    "gdn_fixed_recurrent_state_mib": round(gdn_state_fp32 / 2**20, 2),
    "ratio_kv_over_store_logical": round(TOK * kv_per_token / store_bytes, 1),
    "ratio_kv_over_text": round(TOK * kv_per_token / text_bytes, 1),
    "ratio_text_over_store_logical": round(text_bytes / store_bytes, 3),
    "note": "store_logical_* counts the id/key/directory bits the reference dicts hold (store.py Device/Entry); the physical GDN state layout (bit packing, fp32 vs bf16 heads) is not in store.py and is not measured here.",
}
write_section("capacity", out)
print(json.dumps(out, ensure_ascii=False, indent=1))
