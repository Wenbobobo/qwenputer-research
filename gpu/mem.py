import torch
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f'gpu free {free/2**30:.2f} / total {total/2**30:.2f} GiB')
else:
    print('no gpu')
