"""
This file uploads the LTX-Video model from the Hugging Face Hub to beam.cloud Volume.
"""

from beam import function, Volume, Image, env

if env.is_remote():
    from inference import select_model_files, prepare_models_and_enhancers

VOLUME_PATH = "./ckpts"

with open("./requirements.txt", "w") as f:
    python_packages = [l for l in f.readlines() if not f.startswith("#")]

@function(
    image=Image(python_packages=python_packages),
    memory="32Gi",
    cpu=4,
    secrets=["HF_TOKEN"],
    volumes=[Volume(name="LTX-Video", mount_path=VOLUME_PATH)],
)
def upload():
    model_mode = "ltxv_13B_distilled"
    quantization = "int8"
    transformer_dtype_policy = ""

    # 1. Select the model filename and text encoder filename based on arguments
    _, text_encoder_filename = select_model_files(
        model_mode, quantization, transformer_dtype_policy
    )

    # 2. Prepare model download definitions for text encoder and enhancer
    prepare_models_and_enhancers(text_encoder_filename)

    print("Files uploaded successfully")


if __name__ == "__main__":
    upload()
