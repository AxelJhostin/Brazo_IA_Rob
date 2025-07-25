# =================================================================
# MÓDULO: robot_logic.py
# DESCRIPCIÓN: Contiene toda la lógica de negocio para el control
#              del brazo: detección de pose, cálculo de ángulos y gestos.
# VERSIÓN: 1.2 - Posturas Predefinidas (26/07/2024)
# =================================================================

import mediapipe as mp
import numpy as np
import math
import cv2
import config

class AngleProcessor:
    def __init__(self):
        self.smoothed_angles = {
            'proximidad': 90.0, 'hombro': 90.0, 'codo': 90.0,
            'pitch': 90.0, 'roll': 90.0
        }

    def smooth_and_process(self, raw_angles, distancia_actual, distancia_referencia, target_posture=None):
        """
        Suaviza los ángulos. Si hay una 'target_posture', se moverá hacia ella.
        Si no, usará los 'raw_angles' de la cámara.
        """
        angles_to_process = {}

        if target_posture:
            # MODO POSTURA: El objetivo son los ángulos de la postura predefinida.
            angles_to_process = target_posture
        else:
            # MODO NORMAL: El objetivo son los ángulos de la cámara.
            if distancia_referencia > 0:
                dist_min = distancia_referencia - (distancia_referencia * config.DISTANCE_RANGE)
                dist_max = distancia_referencia + (distancia_referencia * config.DISTANCE_RANGE)
                raw_proximidad = np.interp(distancia_actual, [dist_min, dist_max], [0, 180])
            else:
                raw_proximidad = 90
            raw_angles['proximidad'] = raw_proximidad
            angles_to_process = raw_angles

        # Aplicar filtro de suavizado a cada ángulo
        smoothed_output = {}
        for key, raw_value in angles_to_process.items():
            if key in self.smoothed_angles:
                factor = config.PROXIMITY_FILTER_FACTOR if key == 'proximidad' else config.ANGLE_SMOOTHING_FACTOR
                self.smoothed_angles[key] = (factor * raw_value) + ((1 - factor) * self.smoothed_angles[key])
                smoothed_output[key] = int(self.smoothed_angles[key])
            else:
                smoothed_output[key] = raw_value
        
        return smoothed_output

# --- CLASE POSE DETECTOR (sin cambios) ---
class PoseDetector:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.pose = self.mp_pose.Pose(model_complexity=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.hands = self.mp_hands.Hands(model_complexity=1, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

    def find_pose(self, image): return self.pose.process(image)
    def find_hands(self, image): return self.hands.process(image)
    
    def draw_all_landmarks(self, image, pose_results, hand_results):
        if pose_results.pose_landmarks: 
            self.mp_drawing.draw_landmarks(
                image, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=5),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(180, 180, 180), thickness=3)
            )
            h, w, _ = image.shape
            for idx in [11, 12]:
                if idx < len(pose_results.pose_landmarks.landmark):
                    lm = pose_results.pose_landmarks.landmark[idx]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(image, (cx, cy), 12, config.COLORES['hombro'], -1)
            for idx in [13, 14]:
                if idx < len(pose_results.pose_landmarks.landmark):
                    lm = pose_results.pose_landmarks.landmark[idx]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(image, (cx, cy), 12, config.COLORES['codo'], -1)
        
        if hand_results.multi_hand_landmarks:
            for hand_lm in hand_results.multi_hand_landmarks: 
                self.mp_drawing.draw_landmarks(
                    image, hand_lm, self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=5),
                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(180, 180, 180), thickness=2)
                )

# --- FUNCIONES DE CÁLCULO (sin cambios en su lógica interna) ---
def calcular_angulos_brazo(landmarks, h, w):
    shoulder = [landmarks[12].x * w, landmarks[12].y * h]
    elbow = [landmarks[14].x * w, landmarks[14].y * h]
    wrist = [landmarks[16].x * w, landmarks[16].y * h]
    vec_shoulder_elbow = [elbow[0] - shoulder[0], elbow[1] - shoulder[1]]
    mag_vec = np.linalg.norm(vec_shoulder_elbow)
    ang_hombro = 90
    if mag_vec > 0:
        cos_theta = max(min(np.dot(vec_shoulder_elbow, [0, -1]) / mag_vec, 1), -1)
        ang_hombro = np.degrees(np.arccos(cos_theta))
    vec1 = [shoulder[0] - elbow[0], shoulder[1] - elbow[1]]
    vec2 = [wrist[0] - elbow[0], wrist[1] - elbow[1]]
    mag1, mag2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    ang_codo = 90
    if mag1 > 0 and mag2 > 0:
        cos_theta2 = max(min(np.dot(vec1, vec2) / (mag1 * mag2), 1), -1)
        ang_codo = np.degrees(np.arccos(cos_theta2))
    return {'hombro': ang_hombro, 'codo': ang_codo}, elbow, wrist

def detectar_pinza(hand_landmarks):
    puntos = [(4, 2), (8, 6), (12, 10), (16, 14), (20, 18)]
    flexionados = sum(1 for p1, p2 in puntos if hand_landmarks.landmark[p1].y > hand_landmarks.landmark[p2].y)
    return 1 if flexionados >= 4 else 0

def calcular_distancia_mano(hand_landmarks, punto1=5, punto2=17):
    punto_a = hand_landmarks.landmark[punto1]
    punto_b = hand_landmarks.landmark[punto2]
    return math.sqrt((punto_a.x - punto_b.x)**2 + (punto_a.y - punto_b.y)**2)

def calcular_gestos_mano(hand_landmarks, codo, muneca, distancia_rotacion_ref):
    ma = 90
    try:
        ma_raw = math.degrees(math.atan2(hand_landmarks.landmark[5].y - muneca[1], hand_landmarks.landmark[5].x - muneca[0]) - math.atan2(codo[1] - muneca[1], codo[0] - muneca[0]))
        ma = np.interp(abs(ma_raw), [config.PITCH_INPUT_MIN_ANGLE, config.PITCH_INPUT_MAX_ANGLE], [180, 0])
    except: pass
    mr_limitado = 90
    try:
        distancia_actual_rotacion = calcular_distancia_mano(hand_landmarks, 5, 0)
        if distancia_rotacion_ref > 0:
            variacion_rotacion = distancia_actual_rotacion - distancia_rotacion_ref
            mr_limitado = np.interp(variacion_rotacion, [-config.ROLL_INPUT_RANGE, config.ROLL_INPUT_RANGE], [config.ROLL_OUTPUT_MIN_ANGLE, config.ROLL_OUTPUT_MAX_ANGLE])
        else:
            mr_raw = np.interp(hand_landmarks.landmark[5].x - hand_landmarks.landmark[17].x, [-config.ROLL_INPUT_RANGE, config.ROLL_INPUT_RANGE], [180, 0])
            mr_limitado = np.interp(mr_raw, [0, 180], [config.ROLL_OUTPUT_MIN_ANGLE, config.ROLL_OUTPUT_MAX_ANGLE])
    except: pass
    p = 0
    try: p = detectar_pinza(hand_landmarks)
    except: pass
    return {'pitch': ma, 'roll_raw': mr_limitado, 'pinza': p}

def aplicar_limites_seguros(angulos):
    angulos_limitados = {}
    for eje, valor in angulos.items():
        if eje in config.ANGULOS_SEGUROS:
            min_val, max_val = config.ANGULOS_SEGUROS[eje]
            angulos_limitados[eje] = max(min_val, min(max_val, valor))
        else:
            angulos_limitados[eje] = valor
    return angulos_limitados
