#!/bin/bash
docker run --gpus all --rm --shm-size=1g -e "$(env | grep SLURM_ | xargs)" -it video-diffusion ./test.sh
