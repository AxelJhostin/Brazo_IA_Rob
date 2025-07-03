# =================================================================
# PROYECTO: Control de Brazo Robótico con Visión (6 Ejes)
# VERSIÓN: Final con Panel Mejorado
# =================================================================

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import math
import os

# --- PARÁMETROS ---
BASE_X_MIN_PORCENTAJE = 0.15
BASE_X_MAX_PORCENTAJE = 0.85
ROLL_INPUT_RANGE = 0.22
ROLL_OUTPUT_MIN_ANGLE = 40
ROLL_OUTPUT_MAX_ANGLE = 130
PITCH_INPUT_MIN_ANGLE = 150
PITCH_INPUT_MAX_ANGLE = 210
SMOOTHING_FACTOR = 0.85
GESTURE_BUFFER_SIZE = 15
GESTURE_CONFIRMATION_THRESHOLD = 10

# --- SISTEMA DE SEGURIDAD ---
ANGULOS_SEGUROS = {
    'base': (0, 180),
    'hombro': (30, 160),
    'codo': (20, 170),
    'pitch': (0, 180),
    'roll': (40, 130)
}

# --- FILTRO SUAVIZADOR AVANZADO ---
class SmoothFilter:
    def __init__(self, factor=0.85, buffer_size=7):
        self.factor = factor
        self.buffer = deque(maxlen=buffer_size)
        self.smoothed = None
    
    def update(self, value):
        self.buffer.append(value)
        if self.smoothed is None:
            self.smoothed = value
        else:
            self.smoothed = self.smoothed * self.factor + value * (1 - self.factor)
        return self.smoothed

# --- POSE DETECTOR ---
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
        # Dibujar landmarks con colores personalizados
        if pose_results.pose_landmarks: 
            # Dibujar todos los landmarks en rojo (por defecto)
            self.mp_drawing.draw_landmarks(
                image, 
                pose_results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    color=(0, 0, 255), thickness=2, circle_radius=3  # Rojo
                )
            )
            
            # Sobrescribir puntos específicos con colores personalizados
            h, w, _ = image.shape
            
            # Hombros (turquesa)
            for idx in [11, 12]:  # Hombro izquierdo y derecho
                if idx < len(pose_results.pose_landmarks.landmark):
                    lm = pose_results.pose_landmarks.landmark[idx]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(image, (cx, cy), 8, (255, 255, 0), -1)  # Turquesa
            
            # Codos (naranja)
            for idx in [13, 14]:  # Codo izquierdo y derecho
                if idx < len(pose_results.pose_landmarks.landmark):
                    lm = pose_results.pose_landmarks.landmark[idx]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(image, (cx, cy), 8, (0, 165, 255), -1)  # Naranja
        
        # Manos (color rojo por defecto)
        if hand_results.multi_hand_landmarks:
            for hand_lm in hand_results.multi_hand_landmarks: 
                self.mp_drawing.draw_landmarks(
                    image, 
                    hand_lm, 
                    self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                        color=(0, 0, 255), thickness=2, circle_radius=3  # Rojo
                    )
                )

# --- CÁLCULOS DE ÁNGULOS ---
def calcular_angulos_brazo(landmarks, h, w):
    shoulder = [landmarks[12].x * w, landmarks[12].y * h]
    elbow = [landmarks[14].x * w, landmarks[14].y * h]
    wrist = [landmarks[16].x * w, landmarks[16].y * h]

    # BASE
    base = np.interp(wrist[0], [w * BASE_X_MIN_PORCENTAJE, w * BASE_X_MAX_PORCENTAJE], [180, 0])

    # HOMBRO
    vec_shoulder_elbow = [elbow[0] - shoulder[0], elbow[1] - shoulder[1]]
    eje_vertical = [0, -1]
    mag_vec = np.linalg.norm(vec_shoulder_elbow)
    dot = vec_shoulder_elbow[0] * eje_vertical[0] + vec_shoulder_elbow[1] * eje_vertical[1]

    if mag_vec > 0:
        cos_theta = dot / mag_vec
        cos_theta = max(min(cos_theta, 1), -1)
        ang_hombro = np.degrees(np.arccos(cos_theta))
    else:
        ang_hombro = 90

    # CODO
    vec1 = [shoulder[0] - elbow[0], shoulder[1] - elbow[1]]
    vec2 = [wrist[0] - elbow[0], wrist[1] - elbow[1]]
    dot_product = vec1[0]*vec2[0] + vec1[1]*vec2[1]
    mag1 = np.linalg.norm(vec1)
    mag2 = np.linalg.norm(vec2)

    if mag1 > 0 and mag2 > 0:
        cos_theta2 = dot_product / (mag1 * mag2)
        cos_theta2 = max(min(cos_theta2, 1), -1)
        ang_codo = np.degrees(np.arccos(cos_theta2))
    else:
        ang_codo = 90

    return {'base': base, 'hombro': ang_hombro, 'codo': ang_codo}, elbow, wrist

# --- DETECCIÓN DE GESTOS MEJORADA ---
def detectar_pinza(hand_landmarks):
    puntos = [
        (4, 2), (8, 6), (12, 10), (16, 14), (20, 18)
    ]
    flexionados = sum(
        1 for p1, p2 in puntos 
        if hand_landmarks.landmark[p1].y > hand_landmarks.landmark[p2].y
    )
    return 1 if flexionados >= 4 else 0

def calcular_gestos_mano(hand_landmarks, codo, muneca):
    try:
        ma_raw = math.degrees(math.atan2(hand_landmarks.landmark[5].y - muneca[1], 
                                         hand_landmarks.landmark[5].x - muneca[0]) -
                          math.atan2(codo[1] - muneca[1], codo[0] - muneca[0]))
        ma = np.interp(abs(ma_raw), [PITCH_INPUT_MIN_ANGLE, PITCH_INPUT_MAX_ANGLE], [180, 0])
    except:
        ma = 90

    try:
        p5_x = hand_landmarks.landmark[5].x
        p17_x = hand_landmarks.landmark[17].x
        mr_raw = np.interp(p5_x - p17_x, [-ROLL_INPUT_RANGE, ROLL_INPUT_RANGE], [180, 0])
        mr_limitado = np.interp(mr_raw, [0, 180], [ROLL_OUTPUT_MIN_ANGLE, ROLL_OUTPUT_MAX_ANGLE])
    except:
        mr_limitado = 90

    try:
        p = detectar_pinza(hand_landmarks)
    except:
        p = 0

    return {'pitch': ma, 'roll_raw': mr_limitado, 'pinza': p}

# --- PANEL CON DISEÑO MEJORADO ---
def dibujar_panel(panel, ang_suavizado, pinza_estable, logo_img=None):
    # Fondo gris oscuro para un look profesional
    panel[:] = (40, 40, 40)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.85
    grosor = 2
    
    # Paleta de colores profesional
    color_base = (220, 220, 220)      # Blanco suave
    color_hombro = (255, 255, 0)      # Turquesa
    color_codo = (0, 165, 255)        # Naranja
    color_muneca_pitch = (50, 100, 255)  # Rojo suave
    color_muneca_roll = (100, 230, 100)   # Verde suave
    color_pinza = (200, 200, 200)     # Gris claro
    
    estado_pinza = "CERRADA" if pinza_estable else "ABIERTA"
    
    # Textos alineados con diseño limpio
    textos = [
        ("Base", f"{int(ang_suavizado['base']):>3}", color_base),
        ("Hombro", f"{int(ang_suavizado['hombro']):>3}", color_hombro),
        ("Codo", f"{int(ang_suavizado['codo']):>3}", color_codo),
        ("Muneca Pitch", f"{int(ang_suavizado['pitch']):>3}", color_muneca_pitch),
        ("Muneca Roll", f"{int(ang_suavizado['roll']):>3}", color_muneca_roll),
        ("Pinza", estado_pinza, color_pinza)
    ]

    # Encabezado con diseño minimalista
    cv2.putText(panel, "ESTADO DEL BRAZO", (20, 40), font, 1.1, (240, 240, 240), grosor)
    cv2.line(panel, (15, 60), (430, 60), (80, 80, 80), 2)
    
    # Contenido con espaciado uniforme
    for i, (label, value, color) in enumerate(textos):
        y_pos = 100 + i * 50
        cv2.putText(panel, f"{label}:", (20, y_pos), font, font_scale, color, grosor)
        cv2.putText(panel, value, (220, y_pos), font, font_scale, color, grosor)
    
    # Logo en toda la parte inferior (manteniendo relación de aspecto)
    if logo_img is not None:
        try:
            panel_h, panel_w = panel.shape[:2]
            logo_h, logo_w, _ = logo_img.shape
            
            # Calcular altura máxima disponible para el logo
            max_logo_height = panel_h - 350  # Dejar espacio para los controles
            
            # Calcular proporción de escalado manteniendo relación de aspecto
            scale_factor = min(1.0, max_logo_height / logo_h, panel_w / logo_w)
            new_w = int(logo_w * scale_factor)
            new_h = int(logo_h * scale_factor)
            
            # Redimensionar logo
            logo_resized = cv2.resize(logo_img, (new_w, new_h))
            
            # Posicionar en la parte inferior central
            start_y = panel_h - new_h - 10
            start_x = (panel_w - new_w) // 2
            
            # Solo dibujar si cabe en el panel
            if start_y > 0 and start_x > 0:
                # Manejar transparencia si existe
                if logo_resized.shape[2] == 4:
                    alpha = logo_resized[:, :, 3] / 255.0
                    for c in range(0, 3):
                        panel[start_y:start_y+new_h, start_x:start_x+new_w, c] = (
                            (1 - alpha) * panel[start_y:start_y+new_h, start_x:start_x+new_w, c] + 
                            alpha * logo_resized[:, :, c]
                        )
                else:
                    panel[start_y:start_y+new_h, start_x:start_x+new_w] = logo_resized
                
        except Exception as e:
            print(f"Error dibujando logo: {str(e)}")

# --- VERIFICACIÓN DE SEGURIDAD ---
def verificar_seguridad(angulos):
    for eje, (min_val, max_val) in ANGULOS_SEGUROS.items():
        if eje in angulos:
            valor = angulos[eje]
            if valor < min_val or valor > max_val:
                return False
    return True

# --- MAIN ---
def main():
    # Cargar logo PUCE
    logo_img = None
    try:
        logo_path = "logo_puce.png"
        if os.path.exists(logo_path):
            logo_img = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
            if logo_img is not None:
                print("✅ Logo PUCE cargado")
                # Convertir a BGRA si es necesario
                if logo_img.shape[2] == 3:
                    logo_img = cv2.cvtColor(logo_img, cv2.COLOR_BGR2BGRA)
            else:
                print("⚠️ Error: no se pudo cargar el logo")
        else:
            print("⚠️ Archivo de logo no encontrado")
    except Exception as e:
        print(f"⚠️ Error cargando logo: {str(e)}")
        logo_img = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara")
        return

    detector = PoseDetector()
    gesture_buffer = deque(maxlen=GESTURE_BUFFER_SIZE)
    
    # Filtros suavizadores
    base_filter = SmoothFilter(factor=0.9)
    hombro_filter = SmoothFilter(factor=0.9)
    codo_filter = SmoothFilter(factor=0.9)
    pitch_filter = SmoothFilter(factor=0.85)
    roll_filter = SmoothFilter(factor=0.85)
    panel_width = 450  # Manteniendo la resolución original del panel

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo capturar el frame")
            break
            
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Mantener resolución original
        lienzo = np.zeros((h, w + panel_width, 3), dtype=np.uint8)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results_pose = detector.find_pose(image_rgb)
        results_hands = detector.find_hands(image_rgb)

        ang = {'base': 90, 'hombro': 90, 'codo': 90}
        gestos = {'pitch': 90, 'roll_raw': 90, 'pinza': 0}
        codo, muneca = [0, 0], [0, 0]

        if results_pose.pose_landmarks:
            ang, codo, muneca = calcular_angulos_brazo(results_pose.pose_landmarks.landmark, h, w)
            if results_hands.multi_hand_landmarks:
                for hand in results_hands.multi_hand_landmarks:
                    try:
                        codo_rel = [codo[0] / w, codo[1] / h]
                        muneca_rel = [muneca[0] / w, muneca[1] / h]
                        gestos = calcular_gestos_mano(hand, codo_rel, muneca_rel)
                    except Exception as e:
                        gestos = {'pitch': 90, 'roll_raw': 90, 'pinza': 0}

        gesture_buffer.append(gestos['pinza'])
        pinza_estable = 1 if sum(gesture_buffer) >= GESTURE_CONFIRMATION_THRESHOLD else 0

        # Aplicar suavizado avanzado
        ang_suavizado = {
            'base': base_filter.update(ang['base']),
            'hombro': hombro_filter.update(ang['hombro']),
            'codo': codo_filter.update(ang['codo']),
            'pitch': pitch_filter.update(gestos['pitch']),
            'roll': roll_filter.update(gestos['roll_raw'])
        }

        # Verificar seguridad
        modo_seguro = verificar_seguridad(ang_suavizado)

        detector.draw_all_landmarks(frame, results_pose, results_hands)
        lienzo[0:h, 0:w] = frame
        dibujar_panel(lienzo[:, w:], ang_suavizado, pinza_estable, logo_img)
        
        # Mantener resolución original sin redimensionar ventana
        cv2.imshow("Control de Brazo Robótico", lienzo)
        
        if cv2.waitKey(5) & 0xFF == 27: 
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()