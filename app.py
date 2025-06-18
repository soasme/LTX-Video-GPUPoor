# gunicorn -w 2 -b 0.0.0.0:7860 --timeout 600 app:app
import base64
from io import BytesIO
import logging
import os
import time
import uuid

from flask import Flask, request, jsonify, send_from_directory
from PIL import Image

import inference

app = Flask(__name__)

# Configure logging
logger = logging.getLogger("app")
logging.basicConfig(level=logging.INFO)

# Preload models.
# 1. Select the model filename and text encoder filename based on arguments
model_mode = os.environ.get('MODEL_MODE', 'ltxv_13B_distilled')
quantization = os.environ.get('QUANTIZATION', 'int8')
transformer_dtype_policy = os.environ.get('TRANSFORMER_DTYPE_POLICY', '')
model_filename, text_encoder_filename = inference.select_model_files(
    model_mode, quantization, transformer_dtype_policy
)

# 2. Prepare model download definitions for text encoder and enhancer
inference.prepare_models_and_enhancers(text_encoder_filename)

# 3. Load the model and pipeline
quantize_transformer = False
save_quantized = False
mixed_precision_transformer = False
profile_type_id = int(os.environ.get('PROFILE_TYPE_ID', '1'))
model, _ = inference.load_and_profile_model(
    model_filename,
    model_mode,
    quantize_transformer,
    transformer_dtype_policy,
    mixed_precision_transformer,
    save_quantized,
    profile_type_id,
)
model.pipeline.video_pipeline.set_progress_bar_config(disable=True)

# Ensure outputs directory exists
os.makedirs('outputs', exist_ok=True)

@app.route('/download/<path:filename>', methods=['GET'])
def download_file(filename):
    # Only allow files from the outputs directory
    outputs_dir = os.path.abspath('outputs')
    return send_from_directory(outputs_dir, filename, as_attachment=True)

@app.route('/', methods=['POST'])
def run_inference():
    start_time = time.time()
    data = request.get_json()
    logger.info(f"[POST /] Start time: {start_time:.3f}, Payload: {data}")
    required_fields = [
        'image', 'prompt', 'negative_prompt', 'height', 'width',
        'num_frames', 'frame_rate', 'num_inference_steps'
    ]
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({'error': f'Missing fields: {", ".join(missing)}'}), 400
    try:
        # Decode base64 image to PIL.Image
        image_bytes = base64.b64decode(data['image'])
        pil_image = Image.open(BytesIO(image_bytes)).convert('RGB')
        infer_args = dict(
            prompt=data['prompt'],
            negative_prompt=data['negative_prompt'],
            height=int(data['height']),
            width=int(data['width']),
            video_length=int(data['num_frames']),
            frame_rate=int(data['frame_rate']),
            num_inference_steps=int(data['num_inference_steps']),
            image_start=[pil_image],
            cleanup_model=False,
            model=model,
        )
        output_path = inference.infer(**infer_args)
        # Make the output_path relative to outputs/ for download URL
        outputs_dir = os.path.abspath('outputs')
        abs_output_path = os.path.abspath(output_path)
        if abs_output_path.startswith(outputs_dir):
            rel_path = os.path.relpath(abs_output_path, outputs_dir)
            download_url = request.url_root.rstrip('/') + '/download/' + rel_path
            if os.environ.get('HTTPS', 'false').lower() == 'true':
                download_url = download_url.replace('http://', 'https://')
        else:
            download_url = None
        end_time = time.time()
        logger.info(f"[POST /] End time: {end_time:.3f}, Download URL: {download_url}, Duration: {end_time - start_time:.3f}s")
        return jsonify([{'video': download_url}])
    except Exception as e:
        import traceback; traceback.print_exc()
        logger.error(f"[POST /] Exception: {e}")
        return jsonify([{'error': str(e)}]), 500
