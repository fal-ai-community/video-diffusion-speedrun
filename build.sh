# build script for container; requires HF_HUB_TOKEN
# do NOT run this with pipefail / set exit

pushd /tmp && git clone --branch docker --single-branch https://github.com/fal-ai-community/video-diffusion-speedrun && cd video-diffusion-speedrun
curl -LsSf https://astral.sh/uv/install.sh | sh
DOCKER_BUILDKIT=1 docker build --secret id=hf_token,env=HF_HUB_TOKEN . -t video-diffusion:latest
popd && rm -rf /tmp/video-diffusion-speedrun
docker save -o video-diffusion.tar video-diffusion:latest
