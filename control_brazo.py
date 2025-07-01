# =================================================================
# PROYECTO: Control de Brazo Robótico con Visión (6 Ejes)
# VERSIÓN:  Final con Control 1 a 1 y Resolución Ajustada
# =================================================================

# --- IMPORTACIÓN DE LIBRERÍAS ---
import cv2
import mediapipe as mp
import numpy as np
import math
import serial
import time
from collections import deque

# --- 1. PARÁMETROS DE CONTROL ---
# Se han eliminado los parámetros de calibración para un control más directo.
SMOOTHING_FACTOR = 0.8
SEND_INTERVAL = 0.1
GESTURE_BUFFER_SIZE = 10 
GESTURE_CONFIRMATION_THRESHOLD = 7 
FLEXION_ANGLE_THRESHOLD = 110

# --- 2. CLASE DE AYUDA PARA MEDIAPIPE ---
class PoseDetector:
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
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

def calcular_angulos_brazo(landmarks, h, w):
    hombro = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
    codo = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
    muneca = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].y * h]
    cadera = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].y * h]
    
    # Base sigue siendo mapeada, ya que no hay un ángulo corporal directo para ella.
    h1 = np.interp(muneca[0], [w * 0.15, w * 0.85], [180, 0])
    h2 = calcular_angulo(cadera, hombro, codo)
    c = calcular_angulo(hombro, codo, muneca)
    
    return {'base': h1, 'hombro': h2, 'codo': c}, codo, muneca

def calcular_gestos_mano(hand_landmarks, codo, muneca):
    # Inclinación (Pitch) - AHORA ES UN VALOR DIRECTO 1 a 1
    base_mano_coords = [hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP].x, hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP].y]
    ma = calcular_angulo(codo, muneca, base_mano_coords)

    # Rotación (Roll) - Sigue siendo mapeada por necesidad.
    p5_x = hand_landmarks.landmark[5].x
    p17_x = hand_landmarks.landmark[17].x
    mr_raw = np.interp(p5_x - p17_x, [-0.22, 0.22], [180, 0])

    # Pinza (Abierta/Cerrada)
    puntos_dedos = [(8, 6), (12, 10), (16, 14), (20, 18)]
    dedos_flexionados = sum(1 for p1, p2 in puntos_dedos if hand_landmarks.landmark[p1].y > hand_landmarks.landmark[p2].y)
    p = 1 if dedos_flexionados >= 3 else 0
    
    return {'pitch': ma, 'roll_raw': mr_raw, 'pinza': p}

def dibujar_panel_de_datos(panel, angulos_brazo, gestos_mano, roll_suavizado):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    color_texto = (255, 255, 255)
    grosor = 2
    mano_str = "CERRADA" if gestos_mano['pinza'] == 1 else "ABIERTA"
    textos = {
        "Base (Eje 1)": int(angulos_brazo['base']), "Hombro (Eje 2)": int(angulos_brazo['hombro']),
        "Codo (Eje 3)": int(angulos_brazo['codo']), "Inclinacion (Eje 4)": int(gestos_mano['pitch']),
        "Rotacion (Eje 5)": int(roll_suavizado), "Pinza (Eje 6)": mano_str
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
    # Se eliminó cap.set() para usar la resolución por defecto (más pequeña)
    
    last_send_time = time.time()
    angulo_rotacion_suavizado = 90.0
    gesture_buffer = deque(maxlen=GESTURE_BUFFER_SIZE)
    stable_pinza_state = 0
    
    panel_width = 400

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
        gestos_mano['pinza'] = stable_pinza_state
        dibujar_panel_de_datos(lienzo[:, w:], angulos_brazo, gestos_mano, roll_suavizado)
        
        cv2.imshow('Control Brazo Robótico (6 Ejes)', lienzo)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    print("\nCerrando programa...")
    if arduino:
        arduino.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
