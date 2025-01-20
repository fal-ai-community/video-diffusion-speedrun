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
### install
`uv sync`

`uv pip install git+ssh://git@github.com/fal-ai/lavender-data.git` if you want to use the real dataset

### Run
A `HF_HUB_TOKEN` with read access to `black-forest-labs/FLUX.1-dev` is required

### Single node 8 gpu
`HF_HUB_TOKEN=... ./test.sh`
