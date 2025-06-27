# -----------------------------------------------------------------
# PROYECTO: Control de Brazo Robótico con Visión (6 Ejes)
# VERSIÓN: Final Completa con Comunicación Serial
# -----------------------------------------------------------------

import cv2
import mediapipe as mp
import numpy as np
import math
import serial
import time

# --- 1. CONFIGURACIÓN DE LA COMUNICACIÓN SERIAL ---
try:
    # Reemplaza 'COM4' por el puerto COM correcto de tu placa
    arduino = serial.Serial('COM4', 9600, timeout=1)
    time.sleep(2)
    print("Conexión con Arduino establecida.")
except serial.SerialException as e:
    arduino = None
    print(f"--- ADVERTENCIA: No se pudo conectar con Arduino: {e} ---")
    print("--- El programa continuará en modo visual sin envío de datos. ---")

# --- 2. INICIALIZACIÓN DE MEDIAPIPE ---
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- 3. FUNCIONES Y VARIABLES DE CONTROL ---
def calcular_angulo(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# Variables para suavizado y control de envío
angulo_rotacion_suavizado = 90.0
smoothing_factor = 0.8
send_interval = 0.1
last_send_time = time.time()

# --- 4. BUCLE PRINCIPAL ---
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results_pose = pose.process(image_rgb)
    results_hands = hands.process(image_rgb)

    # Valores por defecto
    h1, h2, c, ma, mr, p = 90, 90, 90, 90, 90, 0
    estado_mano_str = "ABIERTA"
    angulo_rotacion_actual = angulo_rotacion_suavizado

    if results_pose.pose_landmarks:
        try:
            landmarks = results_pose.pose_landmarks.landmark
            
            # Coordenadas
            hombro = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
            codo = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
            muneca = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]
            cadera = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h]
            
            # --- CÁLCULO DE ÁNGULOS ---
            # Hombro Eje 1 (Base/Giro): Mapeo de la posición Y de la muñeca
            h1 = np.interp(muneca[1], [h * 0.1, h * 0.9], [180, 0])
            
            # Hombro Eje 2 (Elevación)
            h2 = calcular_angulo(cadera, hombro, codo)
            
            # Codo
            c = calcular_angulo(hombro, codo, muneca)
            
            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    
                    # Muñeca Adelante/Atrás (Pitch) - ¡NUEVO!
                    base_mano = [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * w, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * h]
                    ma = calcular_angulo(codo, muneca, base_mano)
                    
                    # Muñeca Lados (Roll)
                    p5_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
                    p17_x = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x
                    rotacion_valor = p5_x - p17_x
                    angulo_rotacion_actual = np.interp(rotacion_valor, [-0.15, 0.15], [180, 0])

                    # Pinza (Mano Abierta/Cerrada)
                    puntos_dedos = [(8, 6), (12, 10), (16, 14), (20, 18)]
                    dedos_flexionados = sum(1 for p1, p2 in puntos_dedos if hand_landmarks.landmark[p1].y > hand_landmarks.landmark[p2].y)
                    if dedos_flexionados >= 3:
                        p = 1
                        estado_mano_str = "CERRADA"
                    else:
                        p = 0
                        estado_mano_str = "ABIERTA"

            # Aplicar filtro de suavizado a la rotación
            mr = (angulo_rotacion_suavizado * smoothing_factor) + (angulo_rotacion_actual * (1 - smoothing_factor))
            angulo_rotacion_suavizado = mr

            # --- ENVIAR DATOS A ARDUINO ---
            current_time = time.time()
            if arduino and (current_time - last_send_time > send_interval):
                mensaje = f"<{int(h1)},{int(h2)},{int(c)},{int(ma)},{int(mr)},{p}>\n"
                arduino.write(mensaje.encode('utf-8'))
                last_send_time = current_time

            # --- VISUALIZACIÓN ---
            mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, f"Mano: {estado_mano_str}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Rotacion: {int(mr)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Inclinacion: {int(ma)}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


        except Exception as e:
            pass

    cv2.imshow('Control Brazo Robótico (6 Ejes) - [ESC para Salir]', frame)
    if cv2.waitKey(5) & 0xFF == 27: break

# --- LIMPIEZA FINAL ---
print("\nCerrando programa...")
if arduino:
    arduino.close()
cap.release()
cv2.destroyAllWindows()