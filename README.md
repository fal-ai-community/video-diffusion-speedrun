# OpenVid Diffusion

## Install
### local
`uv sync`

`uv pip install git+ssh://git@github.com/fal-ai/lavender-data.git` if you want to use the real dataset

### Docker
tbd


## Run
A `HF_HUB_TOKEN` with read access to `black-forest-labs/FLUX.1-dev` is required

### Single node 8 gpu
`HF_HUB_TOKEN=... ./test.sh`

---

Streamlit Demo:

```
streamlit run sampling/sample.py
```