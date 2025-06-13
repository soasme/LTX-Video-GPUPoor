import os

from pathlib import Path
import gc

import torch
from mmgp import offload, profile_type
from huggingface_hub import hf_hub_download, snapshot_download
from moviepy import ImageSequenceClip
from ltx_video.ltxv import LTXV
from utils.attention import get_attention_modes, get_supported_attention_modes

attention_modes_installed = get_attention_modes()
attention_modes_supported = get_supported_attention_modes()
text_encoder_quantization = "int8"  # Default quantization type, can be changed to "bf16" or "int8"
enhancer_enabled = True
save_path = "outputs/"

vae_config_choices = [
    ("Auto", 0),
    ("Disabled (faster but may require up to 22 GB of VRAM)", 1),
    ("256 x 256 : If at least 8 GB of VRAM", 2),
    ("128 x 128 : If at least 6 GB of VRAM", 3),
]

def computeList(filename):
    if filename == None:
        return []
    pos = filename.rfind("/")
    filename = filename[pos+1:]
    return [filename]  


def get_ltxv_text_encoder_filename(text_encoder_quantization):
    text_encoder_filename = "ckpts/T5_xxl_1.1/T5_xxl_1.1_enc_bf16.safetensors"
    if text_encoder_quantization =="int8":
        text_encoder_filename = text_encoder_filename.replace("bf16", "quanto_bf16_int8") 
    return text_encoder_filename

def process_files_def(repoId, sourceFolderList, fileList):
    targetRoot = "ckpts/" 
    for sourceFolder, files in zip(sourceFolderList,fileList ):
        if len(files)==0:
            if not Path(targetRoot + sourceFolder).exists():
                snapshot_download(repo_id=repoId,  allow_patterns=sourceFolder +"/*", local_dir= targetRoot)
        else:
            for onefile in files:     
                if len(sourceFolder) > 0: 
                    if not os.path.isfile(targetRoot + sourceFolder + "/" + onefile ):          
                        hf_hub_download(repo_id=repoId,  filename=onefile, local_dir = targetRoot, subfolder=sourceFolder)
                else:
                    if not os.path.isfile(targetRoot + onefile ):          
                        hf_hub_download(repo_id=repoId,  filename=onefile, local_dir = targetRoot)



def get_auto_attention():
    for attn in ["sage2","sage","sdpa"]:
        if attn in attention_modes_supported:
            return attn
    return "sdpa"



transformer_choices = [
    "ckpts/ltxv_0.9.7_13B_dev_bf16.safetensors",
    "ckpts/ltxv_0.9.7_13B_dev_quanto_bf16_int8.safetensors",
    "ckpts/ltxv_0.9.7_13B_distilled_lora128_bf16.safetensors",
]


model_signatures = {"ltxv_13B" : "ltxv_0.9.7_13B_dev",
                    "ltxv_13B_distilled" : "ltxv_0.9.7_13B_distilled",
                    }
finetunes = {}

major, minor = torch.cuda.get_device_capability()
if  major < 8:
    print("Switching to FP16 models when possible as GPU architecture doesn't support optimed BF16 Kernels")
    bfloat16_supported = False
else:
    bfloat16_supported = True

def get_transformer_dtype(model_family, transformer_dtype_policy):
    if len(transformer_dtype_policy) == 0:
        if not bfloat16_supported:
            return torch.float16
        else:
            return torch.bfloat16
    elif transformer_dtype_policy =="fp16":
        return torch.float16
    else:
        return torch.bfloat16
    
def get_model_filename(model_type, quantization ="int8", dtype_policy = ""):
    finetune_def = finetunes.get(model_type, None)
    if finetune_def != None: 
        choices = [ "ckpts/" + os.path.basename(path) for path in finetune_def["URLs"] ]
    else:
        signature = model_signatures[model_type]
        choices = [ name for name in transformer_choices if signature in name]
    if len(quantization) == 0:
        quantization = "bf16"

    model_family =  "ltxv"
    dtype = get_transformer_dtype(model_family, dtype_policy)
    if len(choices) <= 1:
        raw_filename = choices[0]
    else:
        if quantization in ("int8", "fp8"):
            sub_choices = [ name for name in choices if quantization in name]
        else:
            sub_choices = [ name for name in choices if "quanto" not in name]

        if len(sub_choices) > 0:
            dtype_str = "fp16" if dtype == torch.float16 else "bf16"
            new_sub_choices = [ name for name in sub_choices if dtype_str in name]
            sub_choices = new_sub_choices if len(new_sub_choices) > 0 else sub_choices
            raw_filename = sub_choices[0]
        else:
            raw_filename = choices[0]

    if dtype == torch.float16 and not "fp16" in raw_filename and model_family == "wan" and finetune_def == None :
        if "quanto_int8" in raw_filename:
            raw_filename = raw_filename.replace("quanto_int8", "quanto_fp16_int8")
        elif "quanto_bf16_int8" in raw_filename:
            raw_filename = raw_filename.replace("quanto_bf16_int8", "quanto_fp16_int8")
        elif "quanto_mbf16_int8" in raw_filename:
            raw_filename= raw_filename.replace("quanto_mbf16_int8", "quanto_mfp16_int8")
    return raw_filename



def load_ltxv_model(model_filename, base_model_type, quantizeTransformer = False, dtype = torch.bfloat16, VAE_dtype = torch.float32, mixed_precision_transformer = False, save_quantized = False):
    ltxv_model = LTXV(
        model_filepath = model_filename,
        text_encoder_filepath = get_ltxv_text_encoder_filename(text_encoder_quantization),
        dtype = dtype,
        # quantizeTransformer = quantizeTransformer,
        VAE_dtype = VAE_dtype, 
        mixed_precision_transformer = mixed_precision_transformer
    )

    pipeline = ltxv_model.pipeline 
    pipe = {"transformer" : pipeline.video_pipeline.transformer, "vae" : pipeline.vae, "text_encoder" : pipeline.video_pipeline.text_encoder, "latent_upsampler" : pipeline.latent_upsampler}

    return ltxv_model, pipe


####

def infer(
    prompt: str,
    negative_prompt: str = '',
    image_start=None,
    image_end=None,
    video_source=None,
    num_inference_steps: int = 50,
    image_cond_noise_scale: float = 0.15,
    input_media_path=None,
    strength: float = 1.0,
    seed: int = 42,
    height: int = 704,
    width: int = 1216,
    video_length: int = 81,
    frame_rate: int = 30,
    fit_into_canvas: bool = True,
    device=None,
    VAE_tile_size=None,
    model_mode: str = 'ltxv_13B',
    quantization: str = 'int8',
    transformer_dtype_policy: str = '',
    quantize_transformer: bool = False,
    mixed_precision_transformer: bool = False,
    save_quantized: bool = False,
    output_path: str = None,
    profile_type: int = 2
):
    """
    Generate a video using the LTXV model.

    Args:
        prompt (str): Input prompt for video generation.
        negative_prompt (str): Negative prompt.
        image_start: PIL.Image or list of PIL.Image, or None. Start image.
        image_end: PIL.Image or list of PIL.Image, or None. End image.
        video_source: Path to input video, or None.
        num_inference_steps (int): Sampling steps.
        image_cond_noise_scale (float): Image condition noise scale.
        input_media_path: Input media path, or None.
        strength (float): Strength.
        seed (int): Random seed.
        height (int): Video height.
        width (int): Video width.
        video_length (int): Number of frames.
        frame_rate (int): Frame rate.
        fit_into_canvas (bool): Fit into canvas.
        device: Device to use, or None.
        VAE_tile_size: VAE tile size, or None.
        model_mode (str): Model mode.
        quantization (str): Quantization type.
        transformer_dtype_policy (str): Transformer dtype policy.
        quantize_transformer (bool): Quantize transformer.
        mixed_precision_transformer (bool): Mixed precision transformer.
        save_quantized (bool): Save quantized model.
        output_path (str): Path to save the generated video.
        profile_type (int): Profile type (1-5) for offload.profile. Defaults to 2.
    Returns:
        str: Path to the generated video file.
    """
    # 1. Select the model filename and text encoder filename based on arguments
    model_filename = get_model_filename(
        model_mode,
        quantization,
        transformer_dtype_policy
    )
    text_encoder_filename = get_ltxv_text_encoder_filename(text_encoder_quantization)

    # 2. Prepare model download definitions for text encoder and enhancer
    text_encoder_model_def = {
        "repoId" : "DeepBeepMeep/LTX_Video", 
        "sourceFolderList" :  ["T5_xxl_1.1",  ""  ],
        "fileList" : [
            ["added_tokens.json",
            "special_tokens_map.json",
            "spiece.model",
            "tokenizer_config.json"
            ] + computeList(text_encoder_filename),
            ["ltxv_0.9.7_VAE.safetensors",
            "ltxv_0.9.7_spatial_upscaler.safetensors",
            "ltxv_scheduler.json"
            ] + computeList(model_filename) ]   
    }
    enhancer_model_def = {
        "repoId" : "DeepBeepMeep/LTX_Video",
        "sourceFolderList" : [ "Florence2", "Llama3_2"  ],
        "fileList" : [ ["config.json", "configuration_florence2.py", "model.safetensors", "modeling_florence2.py", "preprocessor_config.json", "processing_florence2.py", "tokenizer.json", "tokenizer_config.json"],["config.json", "generation_config.json", "Llama3_2_quanto_bf16_int8.safetensors", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"]  ]
    }

    # 3. Download model files if not present
    if enhancer_enabled:
        process_files_def(**enhancer_model_def)
    process_files_def(**text_encoder_model_def)

    print(f"Using model file: {model_filename}")
    # 4. Load the model and pipeline
    model, pipe = load_ltxv_model(
        model_filename=[model_filename],
        base_model_type=model_mode,
        quantizeTransformer=quantize_transformer,
        dtype=get_transformer_dtype(model_mode, transformer_dtype_policy),
        VAE_dtype=torch.float32,
        mixed_precision_transformer=mixed_precision_transformer,
        save_quantized=save_quantized
    )

    # 5. Set up memory management and offloading
    profile_type_map = {
        1: profile_type.LowRAM_HighVRAM,
        2: profile_type.HighRAM_LowVRAM,
        3: profile_type.LowRAM_HighVRAM,
        4: profile_type.LowRAM_LowVRAM,
        5: profile_type.VerylowRAM_LowVRAM,
    }
    chosen_profile = profile_type_map.get(profile_type, profile_type.HighRAM_LowVRAM)
    offload.profile(pipe, chosen_profile)
    transformer = pipe["transformer"]
    print(f"Model loaded successfully. Using profile_type {profile_type}.")

    # 6. Prepare for inference: disable gradients, clear memory, create output dir
    torch.set_grad_enabled(False) 
    os.makedirs(save_path, exist_ok=True)
    gc.collect()
    torch.cuda.empty_cache()

    # 7. Prepare arguments for the generate method
    # (already unpacked from function signature)

    # 8. Set VAE tile size if not provided (auto-detect based on VRAM)
    if not VAE_tile_size:
        vae_config = 0 # Auto
        device_mem_capacity = torch.cuda.get_device_properties(0).total_memory / 1048576
        vae_precision = "16"
        VAE_tile_size = model.vae.get_VAE_tile_size(
            vae_config,
            device_mem_capacity,
            vae_precision == "32",
        )

    # 9. Set interrupt flag and attention mode
    model._interrupt = False
    attn = get_auto_attention()
    offload.shared_state["_attention"] =  attn

    # 10. Call the generate method of the model to generate the video
    output = model.generate(
        input_prompt=prompt,
        n_prompt=negative_prompt,
        image_start=image_start,
        image_end=image_end,
        input_video=video_source,
        sampling_steps=num_inference_steps,
        image_cond_noise_scale=image_cond_noise_scale,
        input_media_path=input_media_path,
        strength=strength,
        seed=seed,
        height=height,
        width=width,
        frame_num=video_length,
        frame_rate=frame_rate,
        fit_into_canvas=fit_into_canvas,
        device=device,
        VAE_tile_size=VAE_tile_size,
    )

    # 11. Save the output video to disk
    if output_path is None:
        output_path = os.path.join(save_path, f"generated_{seed}.mp4")
    if hasattr(output, 'save'):
        output.save(output_path)
    elif isinstance(output, str):
        # If output is already a file path
        output_path = output
    else:
        # If output is a numpy array or similar, use moviepy to save
        clip = ImageSequenceClip(list(output), fps=frame_rate)
        clip.write_videofile(output_path, codec="libx264")

    print(f"Video saved to {output_path}")

    # 12. Unload model and free memory
    offload.last_offload_obj.unload_all()
    gc.collect()
    torch.cuda.empty_cache()
    
    return output_path

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="LTXV Video Generation Inference")
    parser.add_argument('--prompt', type=str, required=True, help='Input prompt for video generation')
    parser.add_argument('--negative-prompt', type=str, default='', help='Negative prompt')
    parser.add_argument('--image-start', type=str, default=None, help='Path to start image')
    parser.add_argument('--image-end', type=str, default=None, help='Path to end image')
    parser.add_argument('--video-source', type=str, default=None, help='Path to input video')
    parser.add_argument('--num-inference-steps', type=int, default=50, help='Sampling steps')
    parser.add_argument('--image-cond-noise-scale', type=float, default=0.15, help='Image condition noise scale')
    parser.add_argument('--input-media-path', type=str, default=None, help='Input media path')
    parser.add_argument('--strength', type=float, default=1.0, help='Strength')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--height', type=int, default=704, help='Video height')
    parser.add_argument('--width', type=int, default=1216, help='Video width')
    parser.add_argument('--video-length', type=int, default=81, help='Number of frames')
    parser.add_argument('--frame-rate', type=int, default=30, help='Frame rate')
    parser.add_argument('--fit-into-canvas', action='store_true', help='Fit into canvas')
    parser.add_argument('--device', type=str, default=None, help='Device to use')
    parser.add_argument('--VAE-tile-size', type=int, default=None, help='VAE tile size')
    parser.add_argument('--model-mode', type=str, default='ltxv_13B', help='Model mode')
    parser.add_argument('--quantization', type=str, default='int8', help='Quantization type')
    parser.add_argument('--transformer-dtype-policy', type=str, default='', help='Transformer dtype policy')
    parser.add_argument('--quantize-transformer', action='store_true', help='Quantize transformer')
    parser.add_argument('--mixed-precision-transformer', action='store_true', help='Mixed precision transformer')
    parser.add_argument('--save-quantized', action='store_true', help='Save quantized model')
    parser.add_argument('--output-path', type=str, default=None, help='Path to save the generated video')
    parser.add_argument('--profile-type', type=int, default=2, choices=[1,2,3,4,5], help='Profile type for offload.profile (1: LowRAM_HighVRAM, 2: HighRAM_LowVRAM, 3: Balanced, 4: MaxPerformance, 5: MinMemory). Default: 2')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    import urllib.request
    import tempfile
    from PIL import Image

    args = parse_args()
    infer_args = vars(args)
    # Remove None values to avoid passing them as kwargs
    infer_args = {k: v for k, v in infer_args.items() if v is not None}

    temp_files = []
    for key in ["image_start", "image_end"]:
        val = infer_args.get(key)
        if val and isinstance(val, str) and val.startswith("https://"):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(val)[-1] or ".png")
            print(f"Downloading {key} from {val} to {tmp.name}")
            urllib.request.urlretrieve(val, tmp.name)
            infer_args[key] = tmp.name
            temp_files.append(tmp.name)

    # Convert image_start and image_end to PIL.Image if they are file paths
    for key in ["image_start", "image_end"]:
        val = infer_args.get(key)
        if val:
            infer_args[key] = [Image.open(val).convert("RGB")]

    try:
        infer(**infer_args)
    finally:
        for f in temp_files:
            try:
                os.remove(f)
            except Exception as e:
                print(f"Warning: could not remove temp file {f}: {e}")

