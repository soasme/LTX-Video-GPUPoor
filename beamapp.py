from beam import Image, Volume, endpoint, env, Output

if env.is_remote():
    import base64

    from PIL import Image as PILImage

    from inference import *

with open("./requirements.txt", "r") as f:
    python_packages = [l.strip() for l in f.readlines() if not l.startswith("#")]

image = (
    Image(python_version="python3.12")
    .add_python_packages(python_packages)
    .add_commands(
        [
            "apt update && apt install git libgl1 -y",
        ]
    )
    .with_envs("HF_HUB_ENABLE_HF_TRANSFER=1")
)

VOLUME_PATH = "./ckpts"

def load_models():
    # 1. Select the model filename and text encoder filename based on arguments
    model_mode = "ltxv_13B_distilled"
    quantization = "int8"
    transformer_dtype_policy = ""
    quantize_transformer = True
    mixed_precision_transformer = False
    save_quantized = False
    profile_type_id = 1

    model_filename, text_encoder_filename = select_model_files(
        model_mode, quantization, transformer_dtype_policy
    )

    # 2. Prepare model download definitions for text encoder and enhancer
    prepare_models_and_enhancers(text_encoder_filename)

    # 3. Load the model and pipeline
    model, _ = load_and_profile_model(
        model_filename,
        model_mode,
        quantize_transformer,
        transformer_dtype_policy,
        mixed_precision_transformer,
        save_quantized,
        profile_type_id,
    )
    return model

@endpoint(
    name="ltxv-i2v",
    on_start=load_models,
    cpu=4,
    memory="32Gi",
    gpu="RTX4090",
    gpu_count=1,
    image=image,
    volumes=[
        Volume(name="LTX-Video", mount_path=VOLUME_PATH),
    ],
    keep_warm_seconds=60 * 2,
)
def i2v(
    context,
    image,
    prompt,
    negative_prompt,
    height,
    width,
    num_frames,
    frame_rate,
    num_inference_steps,
    timesteps,
    guidance_scale,
    guidance_rescale,
    num_videos_per_prompt,
    seed,
):
    model = context.on_start_value

    if not model:
        raise ValueError("Model is not loaded. Please check the initialization.")
    
    image_bytes = base64.b64decode(image)
    pil_image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")

    path = infer(
        model=model,
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_start=[pil_image],
        image_end=None,
        video_source=None,
        num_inference_steps=num_inference_steps,
        image_cond_noise_scale = 0.15,
        input_media_path=None,
        strength= 1.0,
        seed=seed,
        height=height,
        width=width,
        video_length=num_frames,
        frame_rate=frame_rate,
        fit_into_canvas=True,
        device=None,
        VAE_tile_size=None,
        model_mode="ltxv_13B_distilled",
        quantization="int8",
        transformer_dtype_policy="",
        quantize_transformer=False,
        mixed_precision_transformer=False,
        save_quantized=False,
        output_path=None,
        profile_type_id=1,
    )

    output_file = Output(path=path)
    output_file.save()

    public_url = output_file.public_url(expires=3600)

    return {
        "results": [
            {"video": public_url},
        ],
    }

### Gunicorn Setup