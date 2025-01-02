from flask import Flask, render_template, request, Response, redirect, url_for
import os
import cv2
from ultralytics import YOLO
import supervision as sv
from werkzeug.utils import secure_filename
import pyresearch 

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load YOLO model
model = YOLO("last.pt")

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_processed_video(input_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Unable to open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame
        resized = cv2.resize(frame, (640, 640))

        # Perform detection
        detections = sv.Detections.from_ultralytics(model(resized)[0])

        # Annotate frame
        annotated = sv.BoundingBoxAnnotator().annotate(scene=resized, detections=detections)
        annotated = sv.LabelAnnotator().annotate(scene=annotated, detections=detections)

        # Encode frame for streaming
        _, buffer = cv2.imencode('.jpg', annotated)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return redirect(url_for('index'))

    file = request.files['video']
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(url_for('index'))

    # Save uploaded video
    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(video_path)

    # Redirect to the streaming page
    return redirect(url_for('process_video', video_path=filename))

@app.route('/process/<video_path>')
def process_video(video_path):
    video_full_path = os.path.join(app.config['UPLOAD_FOLDER'], video_path)
    return Response(generate_processed_video(video_full_path), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
