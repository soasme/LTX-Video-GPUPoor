"""
This file uploads the LTX-Video model from the Hugging Face Hub to beam.cloud Volume.
"""

from beam import function, Volume, Image, env

if env.is_remote():
    from inference import get_ltxv_text_encoder_filename, prepare_models_and_enhancers

VOLUME_PATH = "./ckpts"

with open("./requirements.txt", "r") as f:
    python_packages = [l.strip() for l in f.readlines() if not l.startswith("#")]

@function(
    image=Image(python_packages=python_packages),
    memory="32Gi",
    cpu=4,
    secrets=["HF_TOKEN"],
    volumes=[Volume(name="LTX-Video", mount_path=VOLUME_PATH)],
    name="ltxv-upload",
)
def upload():
    # 1. Select the model filename and text encoder filename based on arguments
    text_encoder_quantization = "int8"
    text_encoder_filename = get_ltxv_text_encoder_filename(text_encoder_quantization)(

    # 2. Prepare model download definitions for text encoder and enhancer
    prepare_models_and_enhancers(text_encoder_filename)

    print("Files uploaded successfully")


if __name__ == "__main__":
    upload()
