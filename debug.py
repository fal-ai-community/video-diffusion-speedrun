from contextlib import contextmanager
from IPython import embed

import torch
import torch.distributed as dist
from torch.distributed import device_mesh as tdm
__import__('lovely_tensors').monkey_patch()

def leave():
    dist.destroy_process_group()
    exit()

@contextmanager
def NoZeroInit():
    # Use this as a context manager to suppress zeros_ init.
    # This is useful if you need to fast-debug a full forward pass without zeros.
    Z = torch.nn.init.zeros_
    torch.nn.init.zeros_ = lambda _:0
    yield
    torch.nn.init.zeros_ = Z

def pr0(*a,**k): 0 if dist.get_rank() else print(*a,**k,flush=True)
def printflock(*args, fcntl=__import__('fcntl'), builtins=__import__('builtins'), **kwargs):
    __import__("time").sleep(dist.get_rank()*0.05)
    with open(__file__, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try: builtins.print(f'[{dist.get_rank()}]', *args, **kwargs)
        finally: fcntl.flock(fh, fcntl.LOCK_UN)

@contextmanager
def sequential_execution(group: dist.ProcessGroup):
    rank = dist.get_rank(group)
    for i in range(group.size()):
        if i == rank: yield
        dist.barrier(group)

def is_last_rank(mesh: tdm.DeviceMesh, k: str): return mesh[k].get_local_rank() == mesh[k].size(0) - 1
def force_local(t) -> torch.Tensor: return getattr(t, 'full_tensor', lambda: t)()

def dist_equal(t: torch.Tensor, g: dist.ProcessGroup):
    x = torch.zeros(g.size(), *t.shape, device=t.device, dtype=t.dtype)
    dist.all_gather_into_tensor(x, t[None], group=g)
    return all(torch.equal(x[i], t) for i in range(g.size()))

def distexit(s: str):
    printflock(s)
    dist.destroy_process_group()
    exit()

def assert_rng_equal(g: dist.ProcessGroup):
    if not dist_equal(torch.cuda.get_rng_state(), g):
        printflock(torch.cuda.get_rng_state())
        distexit('unequal! exiting:')

