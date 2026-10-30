import sys, torch
print('python', sys.version.split()[0])
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
    free, total = torch.cuda.mem_get_info()
    print('mem_free_GB', round(free/2**30,2), 'mem_total_GB', round(total/2**30,2))
    print('bf16_ok', torch.tensor([1.0], dtype=torch.bfloat16, device='cuda').item() == 1.0)
import transformers
print('transformers', transformers.__version__)
from transformers import Qwen3_5ForCausalLM
print('qwen3_5 import OK')