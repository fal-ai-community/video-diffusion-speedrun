import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import logging
from tqdm.auto import tqdm
import os
import sys

sys.path.append("/home/ubuntu/simo/cosmos-video-trainer")


from model import (
    DiT,
    timestep_embedding,
)  # You'll need to import these from your models.py
from utils import load_encoders, encode_prompt_with_t5  # Import from your utils.py
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from decoder import get_decoder

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

from decoder import save_tensor_to_mp4

DEFAULT_PROMPT = "The video features a man in a distressed state, lying on the ground with a watch nearby. The man appears to be in a state of emotional turmoil, possibly crying or in pain. The setting is a dimly lit room with a wooden door in the background. The overall style of the video is dramatic and intense, with a focus on the man's emotional state and the surrounding environment. The lighting and composition of the shots suggest a narrative that is centered around the man's experience."


# Cache the model initialization
@st.cache_resource
def init_model(device="cuda", dtype=torch.bfloat16, checkpoint_path=None):
    """Initialize and cache the DiT model."""
    checkpoint_path = Path(checkpoint_path)
    temp_path = checkpoint_path / "temp.pt"

    if not temp_path.exists():
        dcp_to_torch_save(checkpoint_path, temp_path)

    state_dict = torch.load(temp_path, map_location=device)
    with st.spinner("Loading model..."):
        with torch.device("meta"):
            dim = 2048
            model = DiT(
                in_channels=16,
                patch_size=2,
                depth=24,  # Adjust based on your needs
                num_heads=dim // 128,  # Calculated from width/head_dim (3072/64)
                mlp_ratio=4.0,
                cross_attn_input_size=4096,
                hidden_size=dim,  # width parameter
                residual_v=True,
                train_bias_and_rms=False,
            )

        state_dict = {
            k.replace("module.", "")
            .replace("_orig_mod.", ""): v.clone()
            .to(device, dtype)
            for k, v in state_dict.items()
        }
        x = model.load_state_dict(state_dict, strict=True, assign=True)
        logging.info(f"Loaded checkpoint with status: {x}")
        model = model.to(device, dtype)

    return model


# Cache the encoders
@st.cache_resource
def init_encoders(device="cuda", dtype=torch.bfloat16):
    """Initialize and cache the VAE and text encoders."""
    tokenizer, text_encoder = load_encoders(device=device)
    text_encoder = text_encoder.to(device, dtype=dtype)
    return tokenizer, text_encoder


def generate_image(
    prompt: str,
    model,
    vae,
    tokenizer,
    text_encoder,
    device="cuda",
    dtype=torch.bfloat16,
    inference_steps=50,
    cfg_scale=6.0,
    height=512,
    width=512,
    seed=42,
):
    """Generate image from prompt using the DiT model."""
    torch.set_grad_enabled(False)
    RET_INDEX = -8

    # Encode prompts
    prompt_embeds = encode_prompt_with_t5(
        text_encoder, tokenizer, prompt=prompt, device=device, return_index=RET_INDEX
    ).to(dtype)

    negative_embeds = encode_prompt_with_t5(
        text_encoder, tokenizer, prompt="", device=device, return_index=RET_INDEX
    ).to(dtype)

    negative_embeds = torch.zeros_like(negative_embeds)

    # Setup sampling
    dt = 1.0 / inference_steps
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        (1, 16, 16, 2 * (height // 16), 2 * (width // 16)),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    acc_latents = latents.to(dtype=torch.float32)
    dt = torch.tensor([dt] * 1).to(device, dtype=torch.float32).view(1, 1, 1, 1, 1)

    # Initialize progress bar
    progress_bar = st.progress(0)

    # Sampling loop
    for i in range(inference_steps, 0, -1):
        progress = (inference_steps - i) / inference_steps
        progress_bar.progress(progress)

        t_curr = i / inference_steps
        t = i / inference_steps
        t_next = (i - 1) / inference_steps

        alpha = 8.0

        t = t * alpha / (1 + (alpha - 1) * t)
        t_next = t_next * alpha / (1 + (alpha - 1) * t_next)
        dt = t - t_next
        t = torch.tensor([t] * 1).to(device, dtype)
        t_next = torch.tensor([t_next] * 1).to(device, dtype)
        t_curr = torch.tensor([t_curr] * 1).to(device, dtype)

        # Predict noise
        model_output = model(latents, prompt_embeds, t_curr)
        if cfg_scale > 1:
            uncond_output = model(latents, negative_embeds, t_curr)
            model_output = uncond_output + cfg_scale * (model_output - uncond_output)

        # Update latents
        acc_latents = acc_latents + dt * model_output.to(dtype=torch.float32)
        latents = acc_latents.to(dtype=dtype)

    progress_bar.progress(1.0)

    # Decode latents
    # latents = 1 / vae.config.scaling_factor * (acc_latents + vae.config.shift_factor)
    dtype = next(iter(vae.parameters())).dtype
    latents = acc_latents.squeeze(0).to(dtype=dtype)

    print(latents.std())
    # shape must be [T, H, W, C]
    assert len(latents.shape) == 4

    save_tensor_to_mp4(latents, vae, "./output", "test")


def main():
    st.title("DiT Image Generation")

    # Sidebar for configuration
    st.sidebar.header("Generation Settings")
    inference_steps = st.sidebar.slider("Inference Steps", 10, 100, 50)
    cfg_scale = st.sidebar.slider("CFG Scale", 1.0, 20.0, 6.0)
    seed = st.sidebar.number_input("Seed", 0, 1000000, 42)
    height = st.sidebar.number_input("Height", 128, 1024, 512)
    width = st.sidebar.number_input("Width", 128, 1024, 512)

    # Main interface
    prompt = st.text_area("Enter your prompt:", value=DEFAULT_PROMPT, height=100)
    do_enhance = st.checkbox("Enhance prompt", value=True)

    # Initialize models if not already cached
    if "model" not in st.session_state:
        with st.spinner("Loading models... This may take a few minutes."):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16

            checkpoint_path = "/home/ubuntu/simo/cosmos-video-trainer/checkpoints/baseline-large-hs4/67501"

            st.session_state.model = init_model(device, dtype, checkpoint_path)
            st.session_state.tokenizer, st.session_state.text_encoder = init_encoders(
                device, dtype
            )
            st.session_state.vae = get_decoder()

    # Generate button
    if st.button("Generate Image"):
        if not prompt:
            st.warning("Please enter a prompt.")
            return

        with st.spinner("Generating image..."):
            try:
                # if do_enhance:
                #     new_prompt = prompt_enhancer.enhance_prompt(prompt)
                #     st.write(f"Enhanced prompt: {new_prompt}")
                # else:
                new_prompt = prompt

                generate_image(
                    prompt=new_prompt,
                    model=st.session_state.model,
                    vae=st.session_state.vae,
                    tokenizer=st.session_state.tokenizer,
                    text_encoder=st.session_state.text_encoder,
                    inference_steps=inference_steps,
                    cfg_scale=cfg_scale,
                    height=height,
                    width=width,
                    seed=seed,
                )

                # Display the generated video
                st.video("./output/test.mp4")

            except Exception as e:
                st.error(f"An error occurred during image generation: {str(e)}")


if __name__ == "__main__":
    main()


# a tranquil scene of a mountain range shrouded in a veil of fog and clouds. The mountains, cloaked in a dark green hue, rise majestically from the mist, their peaks obscured by a blanket of fog. The sky above is a canvas of overcast gray, adding to the overall somber mood of the scene. Despite the lack of vibrant colors, the scene exudes a sense of serenity and solitude, as if inviting the viewer to lose themselves in the vastness of nature. The perspective from which the photo is taken enhances the grandeur of the mountains, making them appear even more imposing and awe-inspiring. The scene does not contain any discernible text or countable objects, and there are no visible actions taking place. The relative positions of the mountains suggest a vast distance between them, further emphasizing their isolation in the landscape. The scene is devoid of any aesthetic descriptions, focusing solely on the factual elements present within the frame.

# a woman is captured in a moment of tranquility, practicing yoga by the water. She is lying on her stomach on a blue yoga mat, her body forming a straight line from her head to her feet. Her arms are extended straight out in front of her, parallel to the ground. She is wearing a white tank top, which contrasts with the blue of the mat.  The setting is serene and peaceful. The woman is positioned on the left side of the scene, with the vast expanse of the ocean stretching out behind her on the right. The sun is shining brightly, casting a warm glow over the scene and creating a lens flare in the top right corner of the scene. The overall composition of the scene suggests a sense of calm and focus, as the woman engages in her yoga practice amidst the natural beauty of the ocean.


# The video features a man in a distressed state, lying on the ground with a watch nearby. The man appears to be in a state of emotional turmoil, possibly crying or in pain. The setting is a dimly lit room with a wooden door in the background. The overall style of the video is dramatic and intense, with a focus on the man's emotional state and the surrounding environment. The lighting and composition of the shots suggest a narrative that is centered around the man's experience.
