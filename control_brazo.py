# ----------------------------------------------------
# PROYECTO: Control de Brazo Robótico con Visión
# ARCHIVO: control_brazo.py
# FASE 7: Lógica de rotación de muñeca intuitiva
# ----------------------------------------------------

import cv2
import mediapipe as mp
import numpy as np
import math

# --- 1. Inicialización de MediaPipe (Pose y Hands) ---
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5)

# --- 2. Función para Calcular Ángulos (sin cambios) ---
def calcular_angulo(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Inicializar la captura de video
cap = cv2.VideoCapture(0)

# --- VARIABLES PARA EL FILTRO DE SUAVIZADO ---
angulo_rotacion_suavizado = 90.0 
smoothing_factor = 0.8 

# Bucle principal
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(image_rgb)

    # --- 3. Procesamiento y Dibujo ---
    if results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
            
        try:
            landmarks = results_pose.pose_landmarks.landmark
            h, w, _ = frame.shape

            # Coordenadas y ángulos del brazo (sin cambios)
            hombro = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
            codo = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
            muneca = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]
            cadera = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h]
            angulo_hombro = calcular_angulo(cadera, hombro, codo)
            angulo_codo = calcular_angulo(hombro, codo, muneca)

            # --- 4. Detección de la Mano y Rotación ---
            results_hands = hands.process(image_rgb)
            estado_mano = "No Detectada"
            angulo_rotacion_actual = angulo_rotacion_suavizado 

            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))

                    # Lógica para mano abierta/cerrada (sin cambios)
                    punta_indice = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    punta_pulgar = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    base_mano = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    distancia = math.hypot(punta_indice.x - punta_pulgar.x, punta_indice.y - punta_pulgar.y)
                    distancia_referencia = math.hypot(punta_indice.x - base_mano.x, punta_indice.y - base_mano.y)
                    if distancia < distancia_referencia * 0.4: # Umbral ajustado ligeramente
                        estado_mano = "CERRADA"
                    else:
                        estado_mano = "ABIERTA"

                    # --- NUEVA LÓGICA DE ROTACIÓN DE MUÑECA INTUITIVA ---
                    # Usamos la diferencia en X entre la base del dedo índice y la base del meñique
                    # para determinar si la palma o el dorso están al frente.
                    p5_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
                    p17_x = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x
                    
                    # El valor será positivo si la palma está al frente (mano derecha)
                    # y negativo si el dorso está al frente.
                    rotacion_valor = p5_x - p17_x
                    
                    # Mapeamos este valor a un rango de 0-180.
                    # El rango de entrada [-0.15, 0.15] se puede ajustar para cambiar la sensibilidad.
                    # Un rango más grande hará que necesites girar menos la mano.
                    angulo_rotacion_actual = np.interp(rotacion_valor, [-0.15, 0.15], [180, 0])
                    angulo_rotacion_actual = max(0, min(180, angulo_rotacion_actual))

            # --- APLICAR FILTRO DE SUAVIZADO ---
            angulo_rotacion_suavizado = (angulo_rotacion_suavizado * smoothing_factor) + (angulo_rotacion_actual * (1 - smoothing_factor))

            # --- 5. Visualización en Pantalla ---
            cv2.putText(frame, f"Hombro: {int(angulo_hombro)}", tuple(np.multiply(hombro, [1,1]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Codo: {int(angulo_codo)}", tuple(np.multiply(codo, [1,1]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Mano: {estado_mano}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Rotacion: {int(angulo_rotacion_suavizado)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            print(f"\rHombro: {int(angulo_hombro)} | Codo: {int(angulo_codo)} | Mano: {estado_mano} | Rotacion: {int(angulo_rotacion_suavizado)}      ", end="")

        except:
            pass

    cv2.imshow('Control Brazo Robótico - [ESC para Salir]', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

pose.close()
hands.close()
cap.release()
cv2.destroyAllWindows()