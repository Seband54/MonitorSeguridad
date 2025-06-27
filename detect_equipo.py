import torch
import cv2

modelo_equipo = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

def detectar_equipo(frame):
    estado = "Sin detección"
    resultados = modelo_equipo(frame)
    for det in resultados.xyxy[0]:
        x1, y1, x2, y2, conf, cls = det
        label = resultados.names[int(cls)]
        if label in ['person']:
            estado = f"Persona detectada ({conf:.2f})"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return frame, estado
