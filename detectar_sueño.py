import cv2
import mediapipe as mp
import time
import math

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# IDs de landmarks para los ojos (ojo izquierdo y derecho)
OJOS_IZQ = [33, 160, 158, 133, 153, 144]
OJOS_DER = [362, 385, 387, 263, 373, 380]

# Umbral de EAR y duración para detectar sueño
EAR_UMBRAL = 0.2
FRAMES_DORMIDO = 30  # ~1 segundo si el video es de 30fps

contador_frames = 0
estado_actual = "Despierto"

def calcular_EAR(ojo):
    A = math.dist(ojo[1], ojo[5])
    B = math.dist(ojo[2], ojo[4])
    C = math.dist(ojo[0], ojo[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detectar_sueño(frame):
    global contador_frames, estado_actual
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = face_mesh.process(rgb)

    if resultados.multi_face_landmarks:
        landmarks = resultados.multi_face_landmarks[0].landmark
        ojo_izq = [(int(landmarks[i].x * frame.shape[1]), int(landmarks[i].y * frame.shape[0])) for i in OJOS_IZQ]
        ojo_der = [(int(landmarks[i].x * frame.shape[1]), int(landmarks[i].y * frame.shape[0])) for i in OJOS_DER]

        ear_izq = calcular_EAR(ojo_izq)
        ear_der = calcular_EAR(ojo_der)
        ear_prom = (ear_izq + ear_der) / 2.0

        if ear_prom < EAR_UMBRAL:
            contador_frames += 1
        else:
            contador_frames = 0

        if contador_frames >= FRAMES_DORMIDO:
            estado_actual = "Durmiendo"
        else:
            estado_actual = "Despierto"

        # Dibujar estado y EAR
        cv2.putText(frame, f"EAR: {ear_prom:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
        cv2.putText(frame, estado_actual, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255) if estado_actual == "Durmiendo" else (0, 255, 255), 2)

    else:
        estado_actual = "Sin rostro"
        contador_frames = 0
        cv2.putText(frame, estado_actual, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)

    return frame, estado_actual
