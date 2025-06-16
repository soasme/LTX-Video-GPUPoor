from beam import Image, Pod, Volume

with open("./requirements.txt", "r") as f:
    python_packages = [l.strip() for l in f.readlines() if not l.startswith("#")]

with open("./local_requirements.txt", "r") as f:
    local_python_packages = [
        l.strip() for l in f.readlines()
        if not (l.startswith("#") or l.startswith("beam"))]

all_packages = python_packages + local_python_packages

image = (
    Image(python_version="python3.12")
    .add_python_packages(all_packages)
    .add_commands(
        [
            "apt update && apt install git libgl1 -y",
        ]
    )
    .with_envs("HF_HUB_ENABLE_HF_TRANSFER=1")
)

VOLUME_PATH = "./ckpts"

web_server = Pod(
    name="ltxv-i2v-pod-1",
    image=image,
    ports=[7860],
    cpu=1,
    gpu="RTX4090",  # Use a GPU suitable for video processing
    gpu_count=1,
    volumes=[
        Volume(name="LTX-Video", mount_path=VOLUME_PATH),
    ],
    env={
        'HTTPS': 'true',
    },
    keep_warm_seconds=60*5,
    memory=32768,  # 32Gi
    entrypoint=["gunicorn", "-w", "2", "-b", "0.0.0.0:7860", "--timeout", "600", "app:app"],
)

res = web_server.create()

print("✨ Web server hosted at:", res.url)