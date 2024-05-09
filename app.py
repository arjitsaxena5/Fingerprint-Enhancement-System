from flask import Flask, request, jsonify, send_file, send_from_directory
import cv2
import numpy as np
from enhance_model import process_image

app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/enhance', methods=['POST'])
def enhance():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'}), 400

    image_file = request.files['image']
    image_path = 'temp_image.jpg'
    image_file.save(image_path)

    enhanced_image = process_image(image_path)
    enhanced_image_path = 'enhanced_image.jpg'
    cv2.imwrite(enhanced_image_path, enhanced_image)

    return send_file(enhanced_image_path, mimetype='image/jpeg', as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)