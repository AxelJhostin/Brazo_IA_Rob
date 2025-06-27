# ----------------------------------------------------
# PROYECTO: Control de Brazo Robótico con Visión
# ARCHIVO: control_brazo.py
# FASE 3: Cálculo y visualización de ángulos
# ----------------------------------------------------

import cv2
import mediapipe as mp
import numpy as np # Importamos numpy para cálculos matemáticos

# --- 1. Inicialización de MediaPipe Pose (sin cambios) ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# --- 2. Función para Calcular Ángulos ---
def calcular_angulo(a, b, c):
    """Calcula el ángulo entre tres puntos (en grados)."""
    # Convertir los puntos a arrays de numpy
    a = np.array(a)  # Primer punto (ej. Hombro)
    b = np.array(b)  # Vértice (ej. Codo)
    c = np.array(c)  # Punto final (ej. Muñeca)
    
    # Calcular los vectores entre los puntos
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    # Asegurarse que el ángulo esté entre 0 y 180
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Inicializar la captura de video
cap = cv2.VideoCapture(0)

# Bucle principal
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Ignorando cuadro vacío de la cámara.")
        continue

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    # --- 3. Extraer Coordenadas y Calcular Ángulos ---
    if results.pose_landmarks:
        # Dibujar el esqueleto primero
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
            
        # Usar un bloque try-except para evitar errores si un punto no es visible
        try:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape

            # Identificar los puntos clave del brazo DERECHO
            hombro = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
            codo = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
            muneca = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]
            cadera = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h]

            # Calcular los ángulos
            angulo_hombro = calcular_angulo(cadera, hombro, codo)
            angulo_codo = calcular_angulo(hombro, codo, muneca)

            # --- 4. Visualización de los Ángulos ---
            # Mostrar el valor del ángulo del codo sobre la articulación
            cv2.putText(frame, f"Codo: {int(angulo_codo)}", 
                        tuple(np.multiply(codo, [1, 1]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Mostrar el valor del ángulo del hombro sobre la articulación
            cv2.putText(frame, f"Hombro: {int(angulo_hombro)}", 
                        tuple(np.multiply(hombro, [1, 1]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Imprimir los ángulos en la terminal para depuración
            print(f"\rAngulo Hombro: {int(angulo_hombro)} | Angulo Codo: {int(angulo_codo)}      ", end="")

        except Exception as e:
            # Si algo falla (un punto no visible), no hacemos nada
            # print(f"Error: {e}") # Descomentar para ver errores específicos
            pass

    # Mostrar la imagen final
    cv2.imshow('Control Brazo Robótico - [ESC para Salir]', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

# Limpieza final
pose.close()
cap.release()
cv2.destroyAllWindows()