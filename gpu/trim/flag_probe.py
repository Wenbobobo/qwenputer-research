import torch, json
def snap(tag):
    d = {}
    for path in ['torch.backends.cuda.matmul.allow_tf32','torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction','torch.backends.cudnn.allow_tf32','torch.get_float32_matmul_precision()']:
        try: d[path] = str(eval(path))
        except Exception as e: d[path] = 'ERR '+str(e)[:40]
    try:
        d['fp32_precision'] = str(torch.backends.cuda.matmul.fp32_precision)
    except Exception as e:
        d['fp32_precision'] = 'ERR '+str(e)[:40]
    try:
        d['float32_precision'] = str(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
    except Exception as e: d['x']=''
    print(tag, json.dumps(d, ensure_ascii=False))
snap('BEFORE_FLA')
import fla
snap('AFTER_FLA')
print('fla', getattr(fla,'__version__','?'))