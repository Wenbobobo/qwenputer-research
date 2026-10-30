"""Selective FLA for transformers 5.19 Qwen3.5: chip layers use reference torch implementations.
Routes both GDN (chunk/recurrent) and causal_conv1d kernels to their reference torch bodies
for layer_idx < n_chip, while keeping FLA/causal_conv1d for Qwen layers >= n_chip.
"""
import functools
import inspect
import os
import types

NAMES = ('torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule', 
         'causal_conv1d_fn', 'causal_conv1d_update')
_STATE = {}


def _reference(f):
    """Extract the unwrapped torch function and apply kwargs filtering."""
    ref = inspect.unwrap(f)
    params = inspect.signature(ref).parameters
    varkw = any(p.kind == p.VAR_KEYWORD for p in params.values())

    @functools.wraps(ref)
    def call(*a, **k):
        if not varkw:
            k = {n: v for n, v in k.items() if n in params}
        return ref(*a, **k)
    call._ref = ref
    return call


def _impl_name(f):
    """Name/module of the implementation bound in transformers' wrapper."""
    g = f
    while g is not None:
        cv = inspect.getclosurevars(g).nonlocals if hasattr(g, '__code__') else {}
        if 'implementation' in cv:
            impl = cv['implementation']
            return f"{getattr(impl, '__module__', '?')}.{getattr(impl, '__name__', '?')}"
        g = getattr(g, '__wrapped__', None)
    return 'unknown'


def install(n_chip=None, verbose=True):
    """Use torch path for layer_idx < n_chip, FLA for the remaining layers. Idempotent."""
    if _STATE:
        return _STATE['report']
    n_chip = int(os.environ.get('SELECTIVE_FLA_NCHIP', 12) if n_chip is None else n_chip)
    import transformers.models.qwen3_5.modeling_qwen3_5 as M
    fast = {n: getattr(M, n) for n in NAMES}
    ref = {n: _reference(fast[n]) for n in NAMES}
    for n in NAMES:
        r = ref[n]._ref
        assert r.__name__ == n and r.__code__.co_filename == M.__file__, (n, r)
    cls = M.Qwen3_5GatedDeltaNet
    orig = cls.forward
    # Clone only the reference forward's globals: chip calls never mutate module globals,
    # so parallel Qwen calls and CUDA graph capture keep a stable FLA binding.
    body = inspect.unwrap(orig)
    private_globals = dict(body.__globals__)
    private_globals.update(ref)
    chip_forward = types.FunctionType(body.__code__, private_globals, body.__name__,
                                      body.__defaults__, body.__closure__)
    chip_forward.__kwdefaults__ = body.__kwdefaults__
    count = {'ref': 0, 'fast': 0}

    @functools.wraps(orig)
    def forward(self, *a, **k):
        if self.layer_idx < n_chip:
            count['ref'] += 1
            return chip_forward(self, *a, **k)
        count['fast'] += 1
        return orig(self, *a, **k)

    cls.forward = forward
    report = {'n_chip': n_chip, 'impl': {n: _impl_name(fast[n]) for n in NAMES},
              'ref': {n: f"{ref[n]._ref.__module__}.{ref[n]._ref.__name__}" for n in NAMES}}
    _STATE.update(report=report, calls=count, orig=orig, cls=cls)
    if verbose:
        print('[selective_fla]', report, flush=True)
    return report


def calls():
    return dict(_STATE.get('calls', {}))


def uninstall():
    if _STATE:
        _STATE['cls'].forward = _STATE['orig']
        _STATE.clear()
