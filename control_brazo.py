# =================================================================
# PROYECTO: Control de Brazo Robótico con Visión (6 Ejes)
# VERSIÓN:  Final con Sensibilidad Aumentada
# =================================================================

import cv2
import mediapipe as mp
import numpy as np
import serial
import time
import math
from collections import deque

# --- PARÁMETROS ---
BASE_CONTROL_RANGE_PX = 150 
HOMBRO_CONTROL_RANGE_PX = 100
PITCH_SENSITIVITY = 0.4 
# <<-- CORRECCIÓN: Se reduce el rango para hacer la rotación más sensible
ROLL_INPUT_RANGE = 0.18       
FLEXION_ANGLE_THRESHOLD = 110
SMOOTHING_FACTOR = 0.8
SEND_INTERVAL = 0.05
GESTURE_BUFFER_SIZE = 10 
GESTURE_CONFIRMATION_THRESHOLD = 7 

# --- CLASE DE AYUDA PARA MEDIAPIPE ---
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

# --- FUNCIONES DE CÁLCULO ---
def calcular_angulo(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

def calcular_angulos_brazo(landmarks, h, w):
    hombro = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
    codo = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
    muneca = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].y * h]
    
    diferencia_x = codo[0] - hombro[0]
    h1 = np.interp(diferencia_x, [-BASE_CONTROL_RANGE_PX, BASE_CONTROL_RANGE_PX], [180, 0])
    
    diferencia_y = hombro[1] - codo[1]
    h2 = np.interp(diferencia_y, [-HOMBRO_CONTROL_RANGE_PX, HOMBRO_CONTROL_RANGE_PX], [0, 180])

    c = calcular_angulo(hombro, codo, muneca)
    
    return {'base': h1, 'hombro': h2, 'codo': c}, codo, muneca

def calcular_gestos_muneca(hand_landmarks, codo_coords, muneca_coords):
    # Inclinación (Pitch)
    vec_antebrazo = np.array(muneca_coords) - np.array(codo_coords)
    mcp_coords = [hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP].x, hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP].y]
    vec_mano = np.array(mcp_coords) - np.array([muneca_coords[0] / w, muneca_coords[1] / h])
    angle_rad = np.arctan2(vec_mano[1], vec_mano[0]) - np.arctan2(vec_antebrazo[1], vec_antebrazo[0])
    angle_deg = np.degrees(angle_rad)
    if angle_deg > 180: angle_deg -= 360
    if angle_deg < -180: angle_deg += 360
    ma = np.interp(angle_deg, [-90 * PITCH_SENSITIVITY, 90 * PITCH_SENSITIVITY], [180, 0])
    ma = max(0, min(180, ma))

    # Rotación (Roll)
    p5_x = hand_landmarks.landmark[5].x
    p17_x = hand_landmarks.landmark[17].x
    mr_raw = np.interp(p5_x - p17_x, [-ROLL_INPUT_RANGE, ROLL_INPUT_RANGE], [180, 0])
    
    # Pinza (Abierta/Cerrada)
    puntos_dedos = [
        (mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP, mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP, mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP),
        (mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP, mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP, mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP),
        (mp.solutions.hands.HandLandmark.RING_FINGER_MCP, mp.solutions.hands.HandLandmark.RING_FINGER_PIP, mp.solutions.hands.HandLandmark.RING_FINGER_TIP),
        (mp.solutions.hands.HandLandmark.PINKY_MCP, mp.solutions.hands.HandLandmark.PINKY_PIP, mp.solutions.hands.HandLandmark.PINKY_TIP)
    ]
    dedos_flexionados = 0
    for mcp_lm, pip_lm, tip_lm in puntos_dedos:
        mcp = [hand_landmarks.landmark[mcp_lm].x, hand_landmarks.landmark[mcp_lm].y]
        pip = [hand_landmarks.landmark[pip_lm].x, hand_landmarks.landmark[pip_lm].y]
        tip = [hand_landmarks.landmark[tip_lm].x, hand_landmarks.landmark[tip_lm].y]
        if calcular_angulo(mcp, pip, tip) < FLEXION_ANGLE_THRESHOLD:
            dedos_flexionados += 1
    p = 1 if dedos_flexionados >= 3 else 0

    return {'pitch': ma, 'roll_raw': mr_raw, 'pinza_raw': p}

# --- FUNCIÓN DE VISUALIZACIÓN ---
def dibujar_panel_de_datos(panel, angulos_brazo, gestos_mano, roll_suavizado, mano_str):
    font = cv2.FONT_HERSHEY_SIMPLEX; font_scale = 0.9; color_texto = (255, 255, 255); grosor = 2
    textos = {
        "Base (Giro)": int(angulos_brazo['base']),
        "Hombro (Elev.)": int(angulos_brazo['hombro']),
        "Codo (Flex.)": int(angulos_brazo['codo']),
        "Muneca (Pitch)": int(gestos_mano['pitch']),
        "Rotacion (Roll)": int(roll_suavizado),
        "Pinza": mano_str
    }
    cv2.putText(panel, "DATOS DEL ROBOT", (30, 40), font, 1.1, color_texto, grosor)
    cv2.line(panel, (20, 60), (380, 60), color_texto, 1)
    for i, (clave, valor) in enumerate(textos.items()):
        texto = f"{clave}: {valor}"
        posicion = (20, 110 + i * 60)
        cv2.putText(panel, texto, posicion, font, font_scale, color_texto, grosor, cv2.LINE_AA)

# --- FUNCIÓN PRINCIPAL (MAIN) ---
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
    panel_width = 450
    angulo_rotacion_suavizado = 90.0
    gesture_buffer = deque(maxlen=GESTURE_BUFFER_SIZE)
    stable_pinza_state = 0
    
    global w, h

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
        gestos_mano = {'pitch': 90, 'roll_raw': 90, 'pinza_raw': 0}

        if results_pose.pose_landmarks:
            angulos_brazo, codo_coords, muneca_coords = calcular_angulos_brazo(results_pose.pose_landmarks.landmark, h, w)
            
            if results_hands.multi_hand_landmarks:
                for hand_lm in results_hands.multi_hand_landmarks:
                    gestos_mano = calcular_gestos_muneca(hand_lm, codo_coords, muneca_coords)
        
        gesture_buffer.append(gestos_mano['pinza_raw'])
        if sum(gesture_buffer) >= GESTURE_CONFIRMATION_THRESHOLD:
            stable_pinza_state = 1
        elif sum(gesture_buffer) <= (GESTURE_BUFFER_SIZE - GESTURE_CONFIRMATION_THRESHOLD):
            stable_pinza_state = 0

        roll_suavizado = (angulo_rotacion_suavizado * SMOOTHING_FACTOR) + (gestos_mano['roll_raw'] * (1 - SMOOTHING_FACTOR))
        angulo_rotacion_suavizado = roll_suavizado
        
        current_time = time.time()
        if arduino and (current_time - last_send_time > SEND_INTERVAL):
            mensaje = f"<{int(angulos_brazo['base'])},{int(angulos_brazo['hombro'])},{int(angulos_brazo['codo'])},{int(gestos_mano['pitch'])},{int(roll_suavizado)},{stable_pinza_state}>\n"
            arduino.write(mensaje.encode('utf-8'))
            last_send_time = current_time

        detector.draw_all_landmarks(frame, results_pose, results_hands)
        lienzo[0:h, 0:w] = frame
        mano_str = "CERRADA" if stable_pinza_state == 1 else "ABIERTA"
        dibujar_panel_de_datos(lienzo[:, w:], angulos_brazo, gestos_mano, roll_suavizado, mano_str)
        
        cv2.imshow('Control Brazo Robótico - Prueba Ejes 1-6 (Final)', lienzo)
        if cv2.waitKey(5) & 0xFF == 27: break

    print("\nCerrando programa...")
    if arduino: arduino.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
