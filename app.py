from flask import Flask, request, render_template, redirect, url_for, send_file
from ultralytics import YOLO
import os
import time
from PIL import Image

app = Flask(__name__)

# --- FIX 1: Absolute Path Handling ---
# This finds exactly where your app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Automatically create the 'static/uploads' folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# --- FIX 2: Dynamic Model Loading ---
model_path = os.path.join(BASE_DIR, 'best.pt')
model = YOLO(model_path)

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # FIX 3: Unique Timestamps to stop caching/wrong images
        ts = str(int(time.time()))
        uploaded_name = f'uploaded_{ts}.jpg'
        result_name = f'result_{ts}.jpg'
        
        uploaded_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_name)
        file.save(uploaded_path)
        
        # Run AI prediction
        results = model.predict(uploaded_path)
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_name)
        
        detections = []
        for result in results:
            img_array = result.plot() 
            img_rgb = img_array[:, :, ::-1] # BGR to RGB
            img_with_annotations = Image.fromarray(img_rgb)
            img_with_annotations.save(result_path)
            
            for box in result.boxes:
                class_id = int(box.cls[0])
                label = model.names[class_id]
                conf = float(box.conf[0])
                detections.append({'label': label, 'conf': f"{conf*100:.1f}%"})
            
        return render_template('result.html', 
                               uploaded_image=uploaded_name, 
                               result_image=result_name,
                               detections=detections,
                               total_defects=len(detections))

@app.route('/display/<filename>')
def display_image(filename):
    # This route serves the images to your browser
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

if __name__ == '__main__':
    # FIX 4: Disable Debug to stop 'Connection Reset' on Windows
    app.run(host='127.0.0.1', port=5000, debug=False)