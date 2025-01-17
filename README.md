# OpenVid Diffusion

## Docker

#### Build
Run this command with a `HF_HUB_TOKEN` that has access to `black-forest-labs/FLUX.1-dev`:
`DOCKER_BUILDKIT=1 docker build --secret id=hf_token,env=HF_HUB_TOKEN . -t video-diffusion`

#### Run (1node)
Execute small model on a fake dataset:
`docker run --gpus all --rm -it video-diffusion`

#### Run (8node)
TBD

## local
### install
`uv sync`

`uv pip install git+ssh://git@github.com/fal-ai/lavender-data.git` if you want to use the real dataset

### Run
A `HF_HUB_TOKEN` with read access to `black-forest-labs/FLUX.1-dev` is required

### Single node 8 gpu
`HF_HUB_TOKEN=... ./test.sh`
