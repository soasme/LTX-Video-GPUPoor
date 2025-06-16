# gunicorn -w 1 -b 0.0.0.0:7860 app:app
import base64
from io import BytesIO
import os

from flask import Flask, request, jsonify, send_from_directory
from PIL import Image

import inference

app = Flask(__name__)

@app.route('/download/<path:filename>', methods=['GET'])
def download_file(filename):
    # Only allow files from the outputs directory
    outputs_dir = os.path.abspath('outputs')
    return send_from_directory(outputs_dir, filename, as_attachment=True)

@app.route('/', methods=['POST'])
def run_inference():
    data = request.get_json()
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
            image_start=[pil_image]
        )
        output_path = inference.infer(**infer_args)
        # Make the output_path relative to outputs/ for download URL
        outputs_dir = os.path.abspath('outputs')
        abs_output_path = os.path.abspath(output_path)
        if abs_output_path.startswith(outputs_dir):
            rel_path = os.path.relpath(abs_output_path, outputs_dir)
            download_url = request.url_root.rstrip('/') + '/download/' + rel_path
        else:
            download_url = None
        return jsonify([{'video': download_url}])
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify([{'error': str(e)}]), 500
