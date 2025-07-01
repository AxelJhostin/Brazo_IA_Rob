# =================================================================
# PROYECTO: Control de Brazo Robótico con Visión (6 Ejes)
# VERSIÓN:  Final con Panel de Estado Descriptivo
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
BASE_X_MIN_PORCENTAJE = 0.15 
BASE_X_MAX_PORCENTAJE = 0.85
ROLL_INPUT_RANGE = 0.22       
PITCH_INPUT_MIN_ANGLE = 150
PITCH_INPUT_MAX_ANGLE = 210
FLEXION_ANGLE_THRESHOLD = 110
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
    
    h1 = np.interp(muneca[0], [w * BASE_X_MIN_PORCENTAJE, w * BASE_X_MAX_PORCENTAJE], [180, 0])
    h2 = calcular_angulo(cadera, hombro, codo)
    c = calcular_angulo(hombro, codo, muneca)
    
    return {'base': h1, 'hombro': h2, 'codo': c}, codo, muneca

def calcular_gestos_mano(hand_landmarks, codo, muneca):
    ma_raw = calcular_angulo(codo, muneca, [hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y])
    ma = np.interp(ma_raw, [PITCH_INPUT_MIN_ANGLE, PITCH_INPUT_MAX_ANGLE], [180, 0])

    p5_x = hand_landmarks.landmark[5].x
    p17_x = hand_landmarks.landmark[17].x
    mr_raw = np.interp(p5_x - p17_x, [-ROLL_INPUT_RANGE, ROLL_INPUT_RANGE], [180, 0])

    puntos_dedos = [(8, 6), (12, 10), (16, 14), (20, 18)]
    dedos_flexionados = sum(1 for p1, p2 in puntos_dedos if hand_landmarks.landmark[p1].y > hand_landmarks.landmark[p2].y)
    p = 1 if dedos_flexionados >= 3 else 0
    
    # Calcular el área de la mano para el control adelante/atrás
    x_coords = [lm.x for lm in hand_landmarks.landmark]
    y_coords = [lm.y for lm in hand_landmarks.landmark]
    area = (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))
    
    return {'pitch': ma, 'roll_raw': mr_raw, 'pinza': p, 'area': area}

# --- 4. NUEVA FUNCIÓN PARA OBTENER DESCRIPCIONES ---
def get_estado_descriptivo(angulos_brazo, gestos_mano, roll_suavizado):
    """Traduce los ángulos a texto descriptivo."""
    descripciones = {}

    # Base
    if angulos_brazo['base'] > 135: descripciones['base'] = "Girando Izquierda"
    elif angulos_brazo['base'] < 45: descripciones['base'] = "Girando Derecha"
    else: descripciones['base'] = "Base Centrada"

    # Hombro
    if angulos_brazo['hombro'] < 60: descripciones['hombro'] = "Hombro Elevado"
    else: descripciones['hombro'] = "Hombro Bajo"

    # Codo
    if angulos_brazo['codo'] > 150: descripciones['codo'] = "Codo Extendido"
    elif angulos_brazo['codo'] < 70: descripciones['codo'] = "Codo Flexionado"
    else: descripciones['codo'] = "Codo Semi-Flex."

    # Inclinación Muñeca (Pitch)
    if gestos_mano['pitch'] > 135: descripciones['pitch'] = "Muñeca Extendida"
    elif gestos_mano['pitch'] < 45: descripciones['pitch'] = "Muñeca Flexionada"
    else: descripciones['pitch'] = "Muñeca Recta"

    # Rotación Muñeca (Roll)
    if roll_suavizado > 135: descripciones['roll'] = "Rot. Anti-horaria"
    elif roll_suavizado < 45: descripciones['roll'] = "Rot. Horaria"
    else: descripciones['roll'] = "Rot. Centrada"
    
    # Brazo Adelante/Atrás (basado en área de la mano)
    if gestos_mano.get('area', 0) > 0.04: descripciones['profundidad'] = "Brazo Adelante"
    elif gestos_mano.get('area', 0) < 0.02: descripciones['profundidad'] = "Brazo Atrás"
    else: descripciones['profundidad'] = "Brazo Medio"

    return descripciones

def dibujar_info(frame, w, descripciones):
    """Dibuja el panel de estado en la parte derecha de la pantalla."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color_negro = (0, 0, 0)
    color_blanco = (255, 255, 255)
    grosor = 2
    
    for i, (clave, valor) in enumerate(descripciones.items()):
        texto = f"{clave.capitalize()}: {valor}"
        text_size = cv2.getTextSize(texto, font, font_scale, grosor)[0]
        posicion = (w - text_size[0] - 10, 40 * (i + 1))
        
        cv2.putText(frame, texto, posicion, font, font_scale, color_blanco, grosor + 2, cv2.LINE_AA)
        cv2.putText(frame, texto, posicion, font, font_scale, color_negro, grosor, cv2.LINE_AA)

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

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape 
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results_pose = detector.find_pose(image_rgb)
        results_hands = detector.find_hands(image_rgb)
        
        angulos_brazo = {'base': 90, 'hombro': 90, 'codo': 90}
        gestos_mano = {'pitch': 90, 'roll_raw': 90, 'pinza': 0, 'area': 0}

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
        
        # Obtener y mostrar descripciones
        descripciones_estado = get_estado_descriptivo(angulos_brazo, gestos_mano, roll_suavizado)
        dibujar_info(frame, w, descripciones_estado)
        
        # Imprimir en consola
        print(f"\r{descripciones_estado}", end="")
        
        current_time = time.time()
        if arduino and (current_time - last_send_time > SEND_INTERVAL):
            mensaje = f"<{int(angulos_brazo['base'])},{int(angulos_brazo['hombro'])},{int(angulos_brazo['codo'])},{int(gestos_mano['pitch'])},{int(roll_suavizado)},{stable_pinza_state}>\n"
            arduino.write(mensaje.encode('utf-8'))
            last_send_time = current_time

        detector.draw_all_landmarks(frame, results_pose, results_hands)
        
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
