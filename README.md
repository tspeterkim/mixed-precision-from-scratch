# mixed-precision-from-scratch

This is an educational repo that exposes all the details of mixed 
precision training. I apply it to accelerate training on a 2-layer
MLP, using a rewritten matmult call in CUDA (`matmult.cu`) to demonstrate clearly
where the acceleration is coming from.

To compare single precision vs. mixed precision training, run:

```bash
python train.py false
python train.py true
```

to get something like this:
```
$ python train.py false
device: cuda, mixed precision training: False (torch.float32)
model memory: 26.05 MB
act/grad memory: 1100.45 MB
total memory: 1126.50 MB
1: loss 2.327, time: 139.196ms
2: loss 2.237, time: 16.598ms
3: loss 2.175, time: 16.179ms
4: loss 2.117, time: 16.206ms
5: loss 2.058, time: 16.187ms
6: loss 2.006, time: 16.207ms
7: loss 1.948, time: 16.304ms
avg: 16.280ms
$ python train.py true
device: cuda, mixed precision training: True (torch.float16)
model memory: 39.08 MB
act/grad memory: 563.25 MB
total memory: 602.33 MB
1: loss 2.328, time: 170.039ms
2: loss 2.236, time: 8.513ms
3: loss 2.176, time: 8.440ms
4: loss 2.117, time: 8.356ms
5: loss 2.059, time: 8.133ms
6: loss 2.006, time: 8.370ms
7: loss 1.948, time: 8.402ms
avg: 8.369ms
```

Read my blog for more details.
