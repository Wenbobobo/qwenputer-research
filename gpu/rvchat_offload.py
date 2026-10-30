"""Run the released Qwenputer checkpoint with Accelerate dispatch.

This wrapper patches only the loading boundary and the post-load RvChat .to(cuda)
call. The upstream repro source remains untouched.
"""
import sys
import time
from pathlib import Path

import torch
from torch import nn

MODEL = "/mnt/c/Projects/Qcompute/.data/Qwen3.5-4B-Qwenputer-Filesystem"


def install_patches():
    import transformers
    from transformers import Qwen3_5ForCausalLM

    original_from_pretrained = Qwen3_5ForCausalLM.from_pretrained
    def dispatched(cls, *args, **kwargs):
        kwargs.setdefault("device_map", "auto")
        kwargs.setdefault("max_memory", {0: "17GiB", "cpu": "30GiB"})
        kwargs.setdefault("dtype", torch.bfloat16)
        print("[gpu] from_pretrained device_map=auto max_memory=17GiB/cpu30GiB", flush=True)
        t = time.time()
        out = original_from_pretrained.__func__(cls, *args, **kwargs)
        print(f"[gpu] dispatched load {time.time()-t:.1f}s map={getattr(out, 'hf_device_map', None)}", flush=True)
        # Accelerate wraps Module.to at the class level; override this instance too.
        out.to = lambda *a, **k: out
        return out
    Qwen3_5ForCausalLM.from_pretrained = classmethod(dispatched)

    original_to = nn.Module.to
    def guarded_to(self, *args, **kwargs):
        if hasattr(self, "hf_device_map"):
            print("[gpu] suppressed RvChat model.to(): hf_device_map active", flush=True)
            return self
        return original_to(self, *args, **kwargs)
    nn.Module.to = guarded_to


def main():
    install_patches()
    # Import only after patching: RvChat resolves the model class in __init__.
    sys.path.insert(0, "/mnt/c/Projects/Qcompute/repro/qwenputer")
    sys.path.insert(0, "/mnt/c/Projects/Qcompute/repro/qwenputer/experimental/riscv")
    from chipformer.rv import chat
    print(f"[gpu] cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}", flush=True)
    chat.main()

if __name__ == "__main__":
    main()
