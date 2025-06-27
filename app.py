from flask import Flask, render_template, Response, jsonify
import cv2
from detect_equipo import detectar_equipo
from detectar_sueño import detectar_sueño

app = Flask(__name__)
camera = cv2.VideoCapture(0)

estado_equipo = "Desconocido"
estado_sueño = "Desconocido"

def generar_frames():
    global estado_equipo, estado_sueño
    while True:
        success, frame = camera.read()
        if not success:
            break

        frame, estado_equipo = detectar_equipo(frame)
        frame, estado_sueño = detectar_sueño(frame)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generar_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/estado')
def estado():
    return jsonify({
        'equipo': estado_equipo,
        'sueno': estado_sueño
    })

if __name__ == '__main__':
    app.run(debug=True)

