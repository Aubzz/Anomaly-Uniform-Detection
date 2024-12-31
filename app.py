from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit
from uniform_check import UniformCheck
import json

# Flask and SocketIO setup
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
uniform_check = UniformCheck()

@app.route('/')
def index():
    """Render the main HTML page."""
    return render_template('index.html')

def generate_frames():
    """Generate video frames and emit anomaly data via SocketIO."""
    while True:
        frame_bytes, detection_data = uniform_check.get_video_with_anomaly_detection()
        socketio.emit('detection_data', detection_data)  # Send anomaly data to frontend
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    socketio.run(app, debug=True)
