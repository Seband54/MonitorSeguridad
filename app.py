from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from detectar_equipo import detectar_equipo
from detectar_sueño import detectar_sueño

app = Flask(__name__)
estado_equipo = "Desconocido"
estado_sueño = "Desconocido"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/procesar_frame', methods=['POST'])
def procesar_frame():
    global estado_equipo, estado_sueño
    data = request.json['frame']
    frame_data = base64.b64decode(data.split(',')[1])
    np_arr = np.frombuffer(frame_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    frame, estado_equipo = detectar_equipo(frame)
    frame, estado_sueño = detectar_sueño(frame)

    _, buffer = cv2.imencode('.jpg', frame)
    frame_b64 = base64.b64encode(buffer).decode('utf-8')
    return jsonify({
        'frame': f'data:image/jpeg;base64,{frame_b64}',
        'equipo': estado_equipo,
        'sueno': estado_sueño
    })

if __name__ == '__main__':
    app.run(debug=True)
