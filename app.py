# gunicorn -w 1 -b 0.0.0.0:7860 app:app
import base64
from io import BytesIO

from flask import Flask, request, jsonify
from PIL import Image

import inference

app = Flask(__name__)

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
        return jsonify({'output_path': output_path})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
