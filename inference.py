import os
import torch
from pathlib import Path
import gc
from huggingface_hub import hf_hub_download, snapshot_download
from ltx_video.ltxv import LTXV

text_encoder_quantization = "int8"  # Default quantization type, can be changed to "bf16" or "int8"
model_filename = "ckpts/ltxv_0.9.7_13B_distilled_lora128_bf16.safetensors"
enhancer_enabled = True
save_path = "outputs/"

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



text_encoder_filename = get_ltxv_text_encoder_filename(text_encoder_quantization)

transformer_choices = [
    "ckpts/ltxv_0.9.7_13B_dev_bf16.safetensors",
    "ckpts/ltxv_0.9.7_13B_dev_quanto_bf16_int8.safetensors",
    "ckpts/ltxv_0.9.7_13B_distilled_lora128_bf16.safetensors",
]

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
            sub_choices = new_sub_choices if len(new_sub_choices) > 0 : sub_choices
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

def infer(**kwargs):
    # Download the model files if they are not present
    if enhancer_enabled:
        process_files_def(**enhancer_model_def)
    process_files_def(**text_encoder_model_def)

    model, pipe = load_ltxv_model(
        model_filename=get_model_filename(kwargs.get("model_mode", "ltxv_13B"), kwargs.get("quantization", text_encoder_quantization), kwargs.get("transformer_dtype_policy", "")),
        base_model_type=kwargs.get("model_mode", "ltxv_13B"),
        quantizeTransformer=kwargs.get("quantize_transformer", False),
        dtype=get_transformer_dtype(kwargs.get("model_mode", "ltxv_13B"),
                                    kwargs.get("transformer_dtype_policy", "")),
        VAE_dtype=torch.float32,
        mixed_precision_transformer=kwargs.get("mixed_precision_transformer", False),
        save_quantized=kwargs.get("save_quantized", False)
    )

    from mmgp import offload, profile_type
    offload.profile(pipe, profile_type.HighRAM_LowVRAM)

    transformer = pipe["transformer"]
    print("Model loaded successfully.")

    torch.set_grad_enabled(False) 
    os.makedirs(save_path, exist_ok=True)
    gc.collect()
    torch.cuda.empty_cache()

    # Now let's genrate the video
    # Prepare arguments for the generate method
    input_prompt = kwargs.get("prompt", "")
    n_prompt = kwargs.get("negative_prompt", "")
    image_start = kwargs.get("image_start", None)
    image_end = kwargs.get("image_end", None)
    input_video = kwargs.get("video_source", None)
    sampling_steps = kwargs.get("num_inference_steps", 50)
    image_cond_noise_scale = kwargs.get("image_cond_noise_scale", 0.15)
    input_media_path = kwargs.get("input_media_path", None)
    strength = kwargs.get("strength", 1.0)
    seed = kwargs.get("seed", 42)
    height = kwargs.get("height", 704)
    width = kwargs.get("width", 1216)
    frame_num = kwargs.get("video_length", 81)
    frame_rate = kwargs.get("frame_rate", 30)
    fit_into_canvas = kwargs.get("fit_into_canvas", True)
    device = kwargs.get("device", None)
    VAE_tile_size = kwargs.get("VAE_tile_size", None)

    # Call the generate method of the model
    output = model.generate(
        input_prompt=input_prompt,
        n_prompt=n_prompt,
        image_start=image_start,
        image_end=image_end,
        input_video=input_video,
        sampling_steps=sampling_steps,
        image_cond_noise_scale=image_cond_noise_scale,
        input_media_path=input_media_path,
        strength=strength,
        seed=seed,
        height=height,
        width=width,
        frame_num=frame_num,
        frame_rate=frame_rate,
        fit_into_canvas=fit_into_canvas,
        device=device,
        VAE_tile_size=VAE_tile_size,
    )

    # Save the output video
    output_path = os.path.join(save_path, f"generated_{seed}.mp4")
    if hasattr(output, 'save'):
        output.save(output_path)
    elif isinstance(output, str):
        # If output is already a file path
        output_path = output
    else:
        # If output is a numpy array or similar, use moviepy to save
        from moviepy import ImageSequenceClip
        clip = ImageSequenceClip(list(output), fps=frame_rate)
        clip.write_videofile(output_path, codec="libx264")

    print(f"Video saved to {output_path}")

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
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    import urllib.request
    import tempfile

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

    try:
        infer(**infer_args)
    finally:
        for f in temp_files:
            try:
                os.remove(f)
            except Exception as e:
                print(f"Warning: could not remove temp file {f}: {e}")

