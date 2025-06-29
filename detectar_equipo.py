import torch
import cv2

# Cargar ambos modelos
modelo_personas = torch.hub.load('ultralytics/yolov5', 'custom', path='models/yolov5n.pt', force_reload=True)
modelo_cascos = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best.pt', force_reload=True)
print("Clases que detecta el modelo:", modelo_personas.names)
print("Clases que detecta el modelo:", modelo_cascos.names)

def detectar_equipo(frame):
    # Detectar personas
    resultados_personas = modelo_personas(frame)
    detecciones_personas = resultados_personas.pandas().xyxy[0]
    
    # Detectar cascos
    resultados_cascos = modelo_cascos(frame)
    detecciones_cascos = resultados_cascos.pandas().xyxy[0]
    
    personas = []
    cascos = []
    
    # Procesar detecciones de personas
    for _, det in detecciones_personas.iterrows():
        if det['name'] == 'person' and det['confidence'] > 0.5:
            x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
            personas.append({
                'bbox': (x1, y1, x2, y2),
                'tiene_casco': False
            })
    
    # Procesar detecciones de cascos
    for _, det in detecciones_cascos.iterrows():
        if det['confidence'] > 0.5 and det['name'].lower() in ['helmet', 'hardhat', 'casco']:
            x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
            cascos.append((x1, y1, x2, y2))
    
    # Asignar cascos a personas
    for persona in personas:
        px1, py1, px2, py2 = persona['bbox']
        area_persona = (px2 - px1) * (py2 - py1)
        
        for casco in cascos[:]:  # Usamos copia para poder modificar la lista
            hx1, hy1, hx2, hy2 = casco
            centro_casco_x = (hx1 + hx2) / 2
            centro_casco_y = (hy1 + hy2) / 2
            
            # Verificar si el centro del casco está dentro del área superior de la persona
            if (px1 < centro_casco_x < px2 and 
                py1 < centro_casco_y < py1 + (py2 - py1) * 0.4):
                persona['tiene_casco'] = True
                cascos.remove(casco)  # Evitar asignar el mismo casco a múltiples personas
                break
    
    # Dibujar resultados
    personas_con_casco = 0
    for persona in personas:
        x1, y1, x2, y2 = persona['bbox']
        color = (0, 255, 0) if persona['tiene_casco'] else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        texto = "Con casco" if persona['tiene_casco'] else "Sin casco"
        cv2.putText(frame, texto, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if persona['tiene_casco']:
            personas_con_casco += 1
    
    # Determinar estado
    if not personas:
        estado = "Sin personas detectadas"
    elif personas_con_casco == len(personas):
        estado = "Todos con casco"
    else:
        estado = f"{personas_con_casco}/{len(personas)} con casco"
    
    return frame, estado