# OpenVid Diffusion

## Docker

#### Build
Run this command with a `HF_HUB_TOKEN` that has access to `black-forest-labs/FLUX.1-dev`:

`./build.sh`

This builds an image (`video-diffusion:latest`) and saves it to `video-diffusion.tar`.

#### Run (1node)
Execute small model on a fake dataset:
`./docker-wrap.sh`

#### Run (8node)
To load the image on all nodes, run something like:
```bash
srun --nodes 8 --exclusive docker load -i /nfs/path/to/video-diffusion.tar
```
Then, you can execute:
```bash
srun --nodes 8 --exclusive ./docker-wrap.sh
```

## local
You need configured ssh `git@github.com` access to install `lavender-data`.

### install
`uv sync && uv pip install -e .[lavender-data]`.

### Run
A `HF_HUB_TOKEN` with read access to `black-forest-labs/FLUX.1-dev` is required

### Single node 8 gpu
`HF_HUB_TOKEN=... ./test.sh`


##### note on torch 2.6
PT 2.6 is needed because the following script will fail on 2.5.1:
```python
from sys import argv

import torch
from torch._dynamo import config, mark_dynamic
config.dynamic_shapes = True

with torch.device('cuda'):
    x = torch.randn(1, 128, D:=4096)
    m = torch.nn.RMSNorm(D, eps=1e-6) if len(argv)>1 else torch.nn.Linear(D,D)
    m = torch.compile(m, dynamic=None)
    mark_dynamic(x, 1, min=2, max=256)
    o = m(x)
```

`py a.py 1` will error on 2.5.1


##### note on parallelism
this trainer supports simple ddp + fsdp2 (zero2) + cp training.

when CP is enabled, it is always used in conjunction with fsdp2. examples:

###### `--fs 1 --cp 1`
this is the default, and implies all GPUs participate in simple DDP.

###### `--fs 2`
this enables fsdp2 with `reshard_after_forward=False` on pairs of GPUs, and otherwise uses DDP across.

###### `--cp 2`
this enables CP *and* fsdp2 across pairs of GPUs.

between a GPU pair,
* each input sequence will be split in half by `context_parallel`
* each parameter will be sharded by half and regathered every step

###### `--fs 2 --cp 4`
The total fsdp2 factor is `fs * cp = 8`, so the effective FSDP world size will be **8**.

within those 8 GPUs, each sequence will be split into quarter chunks.

