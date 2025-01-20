#!/bin/bash
docker run --gpus all --rm -e "$(env | grep SLURM_ | xargs)" -it video-diffusion ./test.sh