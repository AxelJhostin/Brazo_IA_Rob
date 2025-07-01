# =================================================================
# PROYECTO: Control de Brazo Robótico con Visión (6 Ejes)
# VERSIÓN:  Final con Control Intuitivo por Ejes X/Y
# =================================================================

# --- IMPORTACIÓN DE LIBRERÍAS ---
import cv2
import mediapipe as mp
import numpy as np
import math
import serial
import time
from collections import deque

# --- 1. PARÁMETROS DE CALIBRACIÓN Y LÍMITES ---
# RANGO DEL GIRO BASE (Eje 1) - Controlado por Eje X
BASE_X_MIN_PORCENTAJE = 0.15 
BASE_X_MAX_PORCENTAJE = 0.85

# RANGO DE ELEVACIÓN DEL HOMBRO (Eje 2) - Controlado por Eje Y
HOMBRO_Y_MIN_PORCENTAJE = 0.1
HOMBRO_Y_MAX_PORCENTAJE = 0.9

# RANGO DE ROTACIÓN DE MUÑECA (Roll / Eje 5)
ROLL_INPUT_RANGE = 0.22       
ROLL_OUTPUT_MIN_ANGLE = 40 # Límite mínimo para el servo de rotación
ROLL_OUTPUT_MAX_ANGLE = 130 # Límite máximo para el servo de rotación

# RANGO DE INCLINACIÓN DE MUÑECA (Pitch / Eje 4)
PITCH_INPUT_MIN_ANGLE = 150
PITCH_INPUT_MAX_ANGLE = 210

# UMBRAL PARA MANO CERRADA
FLEXION_ANGLE_THRESHOLD = 110

# PARÁMETROS DE CONTROL
SMOOTHING_FACTOR = 0.8
SEND_INTERVAL = 0.1
GESTURE_BUFFER_SIZE = 10 
GESTURE_CONFIRMATION_THRESHOLD = 7 

# --- 2. CLASE DE AYUDA PARA MEDIAPIPE ---
class PoseDetector:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def find_pose(self, image): return self.pose.process(image)
    def find_hands(self, image): return self.hands.process(image)
    def draw_all_landmarks(self, image, pose_results, hand_results):
        if pose_results.pose_landmarks: self.mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        if hand_results.multi_hand_landmarks:
            for hand_lm in hand_results.multi_hand_landmarks: self.mp_drawing.draw_landmarks(image, hand_lm, self.mp_hands.HAND_CONNECTIONS)

# --- 3. FUNCIONES DE CÁLCULO ---
def calcular_angulo(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

def calcular_angulos_brazo(landmarks, h, w):
    hombro = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
    codo = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
    muneca = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].y * h]
    
    # Eje 1 (Base): Mapeo de la posición X de la muñeca.
    h1 = np.interp(muneca[0], [w * BASE_X_MIN_PORCENTAJE, w * BASE_X_MAX_PORCENTAJE], [180, 0])
    # Eje 2 (Hombro): Mapeo de la posición Y de la muñeca.
    h2 = np.interp(muneca[1], [h * HOMBRO_Y_MAX_PORCENTAJE, h * HOMBRO_Y_MIN_PORCENTAJE], [0, 180])
    # Eje 3 (Codo): Cálculo de ángulo directo.
    c = calcular_angulo(hombro, codo, muneca)
    
    return {'base': h1, 'hombro': h2, 'codo': c}, codo, muneca

def calcular_gestos_mano(hand_landmarks, codo, muneca):
    ma_raw = calcular_angulo(codo, muneca, [hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y])
    ma = np.interp(ma_raw, [PITCH_INPUT_MIN_ANGLE, PITCH_INPUT_MAX_ANGLE], [180, 0])

    p5_x = hand_landmarks.landmark[5].x
    p17_x = hand_landmarks.landmark[17].x
    mr_raw = np.interp(p5_x - p17_x, [-ROLL_INPUT_RANGE, ROLL_INPUT_RANGE], [180, 0])
    mr_limitado = np.interp(mr_raw, [0, 180], [ROLL_OUTPUT_MIN_ANGLE, ROLL_OUTPUT_MAX_ANGLE])

    puntos_dedos = [(8, 6), (12, 10), (16, 14), (20, 18)]
    dedos_flexionados = sum(1 for p1, p2 in puntos_dedos if hand_landmarks.landmark[p1].y > hand_landmarks.landmark[p2].y)
    p = 1 if dedos_flexionados >= 3 else 0
    
    return {'pitch': ma, 'roll_raw': mr_limitado, 'pinza': p}

def dibujar_panel_de_datos(panel, angulos_brazo, gestos_mano, roll_suavizado):
    font = cv2.FONT_HERSHEY_SIMPLEX; font_scale = 0.9; color_texto = (255, 255, 255); grosor = 2
    mano_str = "CERRADA" if gestos_mano['pinza'] == 1 else "ABIERTA"
    textos = {
        "Base (Giro)": int(angulos_brazo['base']), "Hombro (Elev.)": int(angulos_brazo['hombro']),
        "Codo (Flex.)": int(angulos_brazo['codo']), "Muneca (Pitch)": int(gestos_mano['pitch']),
        "Rotacion (Roll)": int(roll_suavizado), "Pinza": mano_str
    }
    cv2.putText(panel, "DATOS DEL ROBOT", (30, 40), font, 1.1, color_texto, grosor)
    cv2.line(panel, (20, 60), (380, 60), color_texto, 1)
    for i, (clave, valor) in enumerate(textos.items()):
        texto = f"{clave}: {valor}"
        posicion = (20, 110 + i * 60)
        cv2.putText(panel, texto, posicion, font, font_scale, color_texto, grosor, cv2.LINE_AA)

# --- 5. FUNCIÓN PRINCIPAL (MAIN) ---
def main():
    try:
        arduino = serial.Serial('COM4', 9600, timeout=1)
        time.sleep(2)
        print("Conexión con Arduino establecida.")
    except Exception as e:
        arduino = None
        print(f"--- ADVERTENCIA: No se pudo conectar con Arduino: {e} ---")

    detector = PoseDetector()
    cap = cv2.VideoCapture(0)
    
    last_send_time = time.time()
    angulo_rotacion_suavizado = 90.0
    gesture_buffer = deque(maxlen=GESTURE_BUFFER_SIZE)
    stable_pinza_state = 0
    
    panel_width = 450

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape 
        lienzo = np.zeros((h, w + panel_width, 3), dtype=np.uint8)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results_pose = detector.find_pose(image_rgb)
        results_hands = detector.find_hands(image_rgb)
        
        angulos_brazo = {'base': 90, 'hombro': 90, 'codo': 90}
        gestos_mano = {'pitch': 90, 'roll_raw': 90, 'pinza': 0}

        if results_pose.pose_landmarks:
            angulos_brazo, codo_coords, muneca_coords = calcular_angulos_brazo(results_pose.pose_landmarks.landmark, h, w)
            if results_hands.multi_hand_landmarks:
                for hand_lm in results_hands.multi_hand_landmarks:
                    codo_rel = [codo_coords[0]/w, codo_coords[1]/h]
                    muneca_rel = [muneca_coords[0]/w, muneca_coords[1]/h]
                    gestos_mano = calcular_gestos_mano(hand_lm, codo_rel, muneca_rel)
        
        gesture_buffer.append(gestos_mano['pinza'])
        if sum(gesture_buffer) >= GESTURE_CONFIRMATION_THRESHOLD: stable_pinza_state = 1
        elif sum(gesture_buffer) <= (GESTURE_BUFFER_SIZE - GESTURE_CONFIRMATION_THRESHOLD): stable_pinza_state = 0

        roll_suavizado = (angulo_rotacion_suavizado * SMOOTHING_FACTOR) + (gestos_mano['roll_raw'] * (1 - SMOOTHING_FACTOR))
        angulo_rotacion_suavizado = roll_suavizado
        
        current_time = time.time()
        if arduino and (current_time - last_send_time > SEND_INTERVAL):
            mensaje = f"<{int(angulos_brazo['base'])},{int(angulos_brazo['hombro'])},{int(angulos_brazo['codo'])},{int(gestos_mano['pitch'])},{int(roll_suavizado)},{stable_pinza_state}>\n"
            arduino.write(mensaje.encode('utf-8'))
            last_send_time = current_time

        detector.draw_all_landmarks(frame, results_pose, results_hands)
        lienzo[0:h, 0:w] = frame
        gestos_mano['pinza'] = stable_pinza_state
        dibujar_panel_de_datos(lienzo[:, w:], angulos_brazo, gestos_mano, roll_suavizado)
        
        cv2.imshow('Control Brazo Robótico (6 Ejes)', lienzo)
        if cv2.waitKey(5) & 0xFF == 27: break

    print("\nCerrando programa...")
    if arduino: arduino.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
