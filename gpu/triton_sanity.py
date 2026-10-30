"""Triton-on-ROCm sanity: vector add + a small matmul-ish reduction, checked against torch."""
import time

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=m) + tl.load(y_ptr + offs, mask=m), mask=m)


@triton.jit
def rowsum_kernel(x_ptr, out_ptr, ncol, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + r * ncol + offs, mask=offs < ncol, other=0.0).to(tl.float32)
    tl.store(out_ptr + r, tl.sum(x, axis=0))


def main():
    print("triton", triton.__version__, "torch", torch.__version__, torch.cuda.get_device_name(0),
          "backend", triton.runtime.driver.active.get_current_target(), flush=True)
    n = 1 << 24
    x = torch.rand(n, device="cuda")
    y = torch.rand(n, device="cuda")
    out = torch.empty_like(x)
    t = time.time()
    add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
    torch.cuda.synchronize()
    print(f"vector add (first call incl. compile) {time.time() - t:.2f} s; exact:",
          bool(torch.equal(out, x + y)), flush=True)
    t = time.time()
    for _ in range(20):
        add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
    torch.cuda.synchronize()
    dt = (time.time() - t) / 20
    print(f"vector add {n} fp32: {dt * 1e3:.3f} ms/iter, {3 * 4 * n / dt / 1e9:.0f} GB/s", flush=True)
    a = torch.randn(4096, 1000, device="cuda", dtype=torch.bfloat16)
    s = torch.empty(4096, device="cuda")
    rowsum_kernel[(4096,)](a, s, 1000, BLOCK=1024)
    torch.cuda.synchronize()
    err = (s - a.float().sum(1)).abs().max().item()
    print(f"bf16 rowsum max abs err vs torch: {err:.2e} ->", "OK" if err < 1e-2 else "BAD")


if __name__ == "__main__":
    main()
