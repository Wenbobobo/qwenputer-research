"""Read installed convolution signatures and source for adapter planning (CPU)."""
import inspect
import fla.modules.convolution as c
import transformers.models.qwen3_5.modeling_qwen3_5 as m
print('FLA', c)
for n in dir(c):
    if 'conv' in n.lower() and callable(getattr(c,n)):
        f = getattr(c,n)
        print(n, inspect.signature(f))
for n in ('causal_conv1d_fn', 'causal_conv1d_update'):
    print(inspect.getsource(inspect.unwrap(getattr(m,n))))
print(inspect.getsource(c.ShortConvolution.forward))
