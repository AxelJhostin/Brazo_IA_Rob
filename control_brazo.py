# =================================================================
# PROYECTO: Control de Brazo Robótico con Visión (6 Ejes)
# VERSIÓN:  Final con Corrección de Inclinación y Puño v3
# =================================================================

# --- IMPORTACIÓN DE LIBRERÍAS ---
import cv2
import mediapipe as mp
import numpy as np
import math
import serial
import time

# --- 1. PARÁMETROS DE CALIBRACIÓN Y LÍMITES ---
# En esta sección puedes ajustar la "sensación" del control sin tocar el resto del código.

# SENSIBILIDAD DEL GIRO BASE (Hombro Eje 1)
BASE_Y_MIN_PORCENTAJE = 0.075 
BASE_Y_MAX_PORCENTAJE = 0.925 

# SENSIBILIDAD DE LA ROTACIÓN DE MUÑECA (Roll)
ROLL_INPUT_RANGE = 0.22       

# SENSIBILIDAD DE LA INCLINACIÓN DE MUÑECA (Pitch) - LÓGICA CORREGIDA
# Define el rango del ángulo de tu muñeca (flexión/extensión) que se mapeará a 0-180.
# Un rango más pequeño hará el control más sensible.
PITCH_INPUT_MIN_ANGLE = 150 # Ángulo cuando la muñeca está en máxima extensión
PITCH_INPUT_MAX_ANGLE = 210 # Ángulo cuando la muñeca está en máxima flexión

# UMBRAL PARA MANO CERRADA
FLEXION_ANGLE_THRESHOLD = 110

# PARÁMETROS DE CONTROL
SMOOTHING_FACTOR = 0.8
SEND_INTERVAL = 0.1

# --- 2. CLASE DE AYUDA PARA MEDIAPIPE ---
class PoseDetector:
    """Clase que maneja la inicialización y uso de los modelos de MediaPipe."""
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def find_pose(self, image):
        return self.pose.process(image)

    def find_hands(self, image):
        return self.hands.process(image)

    def draw_all_landmarks(self, image, pose_results, hand_results):
        if pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

# --- 3. FUNCIONES DE CÁLCULO ---
def calcular_angulo(a, b, c):
    """Calcula el ángulo entre tres puntos (vértice en 'b') en grados."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

def calcular_angulos_brazo(landmarks, h, w):
    """Calcula los ángulos para la base, el hombro y el codo."""
    hombro = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
    codo = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
    muneca = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].y * h]
    cadera = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].y * h]
    
    h1 = np.interp(muneca[1], [h * BASE_Y_MIN_PORCENTAJE, h * BASE_Y_MAX_PORCENTAJE], [180, 0])
    h2 = calcular_angulo(cadera, hombro, codo)
    c = calcular_angulo(hombro, codo, muneca)
    
    return {'base': h1, 'hombro': h2, 'codo': c}, codo, muneca

def calcular_gestos_mano(hand_landmarks, codo, muneca, h, w):
    """Calcula los ángulos para la inclinación y rotación de la muñeca, y el estado de la pinza."""
    # --- Inclinación (Pitch) - LÓGICA DE FLEXIÓN/EXTENSIÓN ---
    # Usamos el codo, la muñeca y la base de los dedos para un cálculo preciso.
    base_mano = [hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP].x * w, hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP].y * h]
    ma_raw = calcular_angulo(codo, muneca, base_mano)
    ma = np.interp(ma_raw, [PITCH_INPUT_MIN_ANGLE, PITCH_INPUT_MAX_ANGLE], [180, 0]) # Mapeo corregido

    # Rotación (Roll)
    p5_x = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP].x
    p17_x = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_MCP].x
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
        angulo_flexion = calcular_angulo(mcp, pip, tip)
        if angulo_flexion < FLEXION_ANGLE_THRESHOLD:
            dedos_flexionados += 1
    
    p = 1 if dedos_flexionados >= 3 else 0
    
    return {'pitch': ma, 'roll_raw': mr_raw, 'pinza': p}

def dibujar_info(frame, mano_str, rotacion, inclinacion):
    """Dibuja el texto informativo sobre la ventana de video."""
    cv2.putText(frame, f"Mano: {mano_str}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Rotacion: {int(rotacion)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Inclinacion: {int(inclinacion)}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


# --- 4. FUNCIÓN PRINCIPAL (MAIN) ---
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

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape 
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results_pose = detector.find_pose(image_rgb)
        results_hands = detector.find_hands(image_rgb)
        
        angulos_brazo = {'base': 90, 'hombro': 90, 'codo': 90}
        gestos_mano = {'pitch': 90, 'roll_raw': 90, 'pinza': 0}

        if results_pose.pose_landmarks:
            angulos_brazo, codo_coords, muneca_coords = calcular_angulos_brazo(results_pose.pose_landmarks.landmark, h, w)

            if results_hands.multi_hand_landmarks:
                for hand_lm in results_hands.multi_hand_landmarks:
                    gestos_mano = calcular_gestos_mano(hand_lm, codo_coords, muneca_coords, h, w)

        roll_suavizado = (angulo_rotacion_suavizado * SMOOTHING_FACTOR) + (gestos_mano['roll_raw'] * (1 - SMOOTHING_FACTOR))
        angulo_rotacion_suavizado = roll_suavizado
        
        current_time = time.time()
        if arduino and (current_time - last_send_time > SEND_INTERVAL):
            mensaje = f"<{int(angulos_brazo['base'])},{int(angulos_brazo['hombro'])},{int(angulos_brazo['codo'])},{int(gestos_mano['pitch'])},{int(roll_suavizado)},{gestos_mano['pinza']}>\n"
            arduino.write(mensaje.encode('utf-8'))
            last_send_time = current_time

        detector.draw_all_landmarks(frame, results_pose, results_hands)
        mano_str = "CERRADA" if gestos_mano['pinza'] == 1 else "ABIERTA"
        dibujar_info(frame, mano_str, roll_suavizado, gestos_mano['pitch'])

        cv2.imshow('Control Brazo Robótico (6 Ejes) - [ESC para Salir]', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    print("\nCerrando programa...")
    if arduino:
        arduino.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
