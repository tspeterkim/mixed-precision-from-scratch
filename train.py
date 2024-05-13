import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def cmp(s, a, b): # utility function to compare tensors
    ex = torch.all(a == b).item()
    app = torch.allclose(a, b)
    maxdiff = (a - b).abs().max().item()
    print(f'{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
MP = sys.argv[1] == 'true'
dtype = torch.float16 if MP else torch.float32
print(f'device: {device}, mixed precision training: {MP} ({dtype})')
scale = 128 if MP else 1 # loss scaling

# load mixed precision training CUDA kernels
mpt = load(name='mixed_precision_training',
           sources=['main.cpp', 'matmult.cu'],
           extra_cuda_cflags=['-O2', '-lcublas'])

# define model
batch_size = n = 8192 # n for convenience
g = torch.Generator(device=device).manual_seed(42) # for reproducibility
n_embd = 784
n_hidden = 8192
num_classes = 10

# master weights
# linear layer 1
a, b = - ((1/n_embd) ** 0.5), ((1/n_embd) ** 0.5) # kaiming uniform init
m_W1 = (b-a) * torch.rand((n_embd, n_hidden), generator=g, dtype=torch.float32, device=device) + a
m_b1 = (b-a) * torch.rand(n_hidden, generator=g, dtype=torch.float32, device=device) + a

# linear layer 2
a, b = - ((1/n_hidden) ** 0.5), ((1/n_hidden) ** 0.5)
m_W2 = (b-a) * torch.rand((n_hidden, num_classes), generator=g, dtype=torch.float32, device=device) + a
m_b2 = (b-a) * torch.rand(num_classes, generator=g, dtype=torch.float32, device=device) + a

parameters = [m_W1, m_b1, m_W2, m_b2] # updating fp32 master weights

# allocate activations, gradients memory on global DRAM
# doing this to avoid allocating memory inside the CUDA kernels (unnecessary overhead)
a1 = torch.empty((n, n_hidden), dtype=dtype, device=device)
z1 = torch.empty_like(a1)
logits = torch.empty((n, num_classes), dtype=dtype, device=device)
dlogits = torch.empty_like(logits)
dz1 = torch.empty_like(a1)
dW2 = torch.empty_like(m_W2, dtype=dtype)
db2 = torch.empty_like(m_b2, dtype=dtype)
da1 = torch.empty_like(a1)
dW1 = torch.empty_like(m_W1, dtype=dtype)
db1 = torch.empty_like(m_b1, dtype=dtype)

intermediates = [a1, z1, logits, dlogits, dz1, da1]

# calculate memory consumption
mem_model = 0
for p in parameters:
    mem_model += p.element_size() * p.nelement()
    if MP: # for 16-bit copies for forward/backward
        mem_model += p.element_size()/2 * p.nelement()
print(f'model memory: {mem_model / 1e6:.2f} MB')

mem_rest = 0
for p in parameters + intermediates:
    mem_rest += p.element_size() * p.nelement()
print(f'act/grad memory: {mem_rest / 1e6:.2f} MB')
print(f'total memory: {(mem_model+mem_rest) / 1e6:.2f} MB')

# load mnist
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dloader = DataLoader(dset, batch_size=n, shuffle=False, pin_memory=True)

timings = []
for i, (x,y) in enumerate(dloader):

    t0 = time.time()
    x, y = x.to(dtype).to(device), y.to(device)

    if len(y) != n: # hacky way to ignore the last batch
        break

    # 1. forward pass
    x = x.view(-1, n_embd) # flatten 2d img to 1d
    W1, b1 = m_W1.half() if MP else m_W1, m_b1.half() if MP else m_b1 # use fp16 weight copies
    a1 = b1.repeat((n,1)) # set a1 as biases for cublas GEMM
    mpt.matmult(x, W1, a1, False, False, 1.0, 1.0) # a1 = x @ W1 + b1
    # cmp('a1', a1, x @ W1 + b1)

    z1 = F.relu(a1) # (n, n_hidden)

    W2, b2 = m_W2.half() if MP else m_W2, m_b2.half() if MP else m_b2 # use fp16 weight copies
    logits = b2.repeat((n,1)) # set logits as biases for GEMM

    mpt.matmult(z1, W2, logits, False, False, 1.0, 1.0) # logits = z1 @ W2 + b2
    # cmp('logits', logits, z1 @ W2 + b2)

    loss = F.cross_entropy(logits, y)

    # 2. manual backward pass. Kudos to Andrej, for making me a backprop ninja
    dlogits = F.softmax(logits, 1, dtype=torch.float32) # cast logits to fp32 before softmax
    dlogits[range(n), y] -= 1
    dlogits /= n

    # loss scaling
    # note: we multiply dlogits, instead of loss, by scale. This is because
    #   we are doing backprop manually. The first gradient is dlogits, and the
    #   scale will propogate through all gradients.
    dlogits *= scale
    dlogits = dlogits.to(torch.float16)

    mpt.matmult(dlogits, W2, dz1, False, True, 1.0, 0.0) # dz1 = dlogits @ W2.T
    # cmp('dz1', dz1, dlogits @ W2.T)

    mpt.matmult(z1, dlogits, dW2, True, False, 1.0, 0.0) # dW2 = z1.T @ dlogits
    # cmp('dW2', dW2, z1.T @ dlogits)

    db2 = dlogits.sum(0)
    da1 = dz1 * (a1 > 0).to(dtype)
    mpt.matmult(x, da1, dW1, True, False, 1.0, 0.0) # dW1 = x.T @ da1
    # cmp('dW1', dW1, x.T @ da1)

    db1 = da1.sum(0)

    # 3. SGD update
    grads = [dW1, db1, dW2, db2]
    lr = 0.01
    for p, grad in zip(parameters, grads):
        p.data += -lr * (grad.to(torch.float32) / scale) # cast grad to fp32 before un-scale

    t1 = time.time()
    print(f'{i+1:2d}: loss {loss.item():.3f}, time: {(t1-t0)*1000:.3f}ms')
    timings.append(t1-t0)

print(f'avg: {np.mean(timings[1:])*1000:.3f}ms') # ignore first as outlier
