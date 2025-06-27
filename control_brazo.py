# ----------------------------------------------------
# PROYECTO: Control de Brazo Robótico con Visión
# ARCHIVO: control_brazo.py
# FASE 4: Detección de mano abierta/cerrada
# ----------------------------------------------------

import cv2
import mediapipe as mp
import numpy as np
import math # Importamos math para cálculos de distancia

# --- 1. Inicialización de MediaPipe (Pose y Hands) ---
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands # << NUEVO: Inicializamos el detector de manos
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

hands = mp_hands.Hands( # << NUEVO: Creamos el objeto de manos
    static_image_mode=False,
    max_num_hands=1, # Solo nos interesa una mano
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

            # Coordenadas del brazo
            hombro = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
            codo = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
            muneca = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]
            cadera = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h]

            # Cálculo de ángulos
            angulo_hombro = calcular_angulo(cadera, hombro, codo)
            angulo_codo = calcular_angulo(hombro, codo, muneca)

            # Visualización de ángulos
            cv2.putText(frame, f"Codo: {int(angulo_codo)}", tuple(np.multiply(codo, [1, 1]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Hombro: {int(angulo_hombro)}", tuple(np.multiply(hombro, [1, 1]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # --- 4. Detección de la Mano ---
            # Procesamos la imagen con el detector de manos
            results_hands = hands.process(image_rgb)
            estado_mano = "No Detectada"

            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    # Dibujar la mano
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                    )

                    # Calcular si la mano está abierta o cerrada
                    # Usaremos la distancia vertical entre la punta del índice y la base del pulgar
                    punta_indice = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    punta_pulgar = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    base_mano = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

                    # Calculamos la distancia entre el pulgar y el índice
                    distancia = math.hypot(punta_indice.x - punta_pulgar.x, punta_indice.y - punta_pulgar.y)
                    
                    # Comparamos con la distancia entre la muñeca y la punta del índice para normalizar
                    distancia_referencia = math.hypot(punta_indice.x - base_mano.x, punta_indice.y - base_mano.y)
                    
                    # Si la distancia entre los dedos es pequeña, está cerrada.
                    if distancia < distancia_referencia * 0.3: # El 0.3 es un umbral que puedes ajustar
                        estado_mano = "CERRADA"
                    else:
                        estado_mano = "ABIERTA"

            # Mostramos el estado de la mano en la pantalla
            cv2.putText(frame, f"Mano: {estado_mano}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Imprimir todo en la terminal
            print(f"\rAngulo Hombro: {int(angulo_hombro)} | Angulo Codo: {int(angulo_codo)} | Mano: {estado_mano}      ", end="")

        except:
            pass

    cv2.imshow('Control Brazo Robótico - [ESC para Salir]', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

pose.close()
hands.close() # << NUEVO: Cerramos el detector de manos
cap.release()
cv2.destroyAllWindows()