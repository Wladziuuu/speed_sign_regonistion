#export OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_flags;tcp"
from ultralytics import YOLO
import cv2 as cv
from flask import Flask, Response
import threading
import time

app = Flask(__name__)


global_frame = None
frame_lock = threading.Lock()
running = True


def generate_frames():
    global global_frame
    while running:

        with frame_lock:
            if global_frame is None:
                time.sleep(0.1)
                continue
            frame_copy = global_frame.copy()
        
        # Convert to JPEG
        ret, buffer = cv.imencode('.jpg', frame_copy)
        if not ret:
            continue
            
        # Yield the frame in the format expected by Flask
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def start_server():
    app.run(host='0.0.0.0', port=3768, debug=False, threaded=True)

print("Inicjalizacja strumienia RTSP")
vcap = cv.VideoCapture("rtsp://192.168.1.13:1935/")
print("Strumień RTSP zainicjalizowany")
detector = YOLO("best.pt", task="predict")
print("Model YOLO zainicjalizowany")


server_thread = threading.Thread(target=start_server)
server_thread.daemon = True
server_thread.start()

print("Streaming video to http://localhost:3768")
print("Press Ctrl+C to exit")

try:
    ret = True
    while ret and running:
        ret, frame = vcap.read()
        
        if not ret:
            break
            
        detections = detector(frame, stream=True, vid_stride=1)
        
        for detection in detections:
            for bbox in detection.boxes:
                x1, y1, x2, y2 = bbox.xyxy[0]  
                

                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                

                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        with frame_lock:
            global_frame = frame
            
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
            
except KeyboardInterrupt:
    print("Stopping the application...")
    running = False
    
finally:
    vcap.release()
    cv.destroyAllWindows()