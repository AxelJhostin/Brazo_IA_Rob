# =================================================================
# MÓDULO: robot_logic.py
# VERSIÓN: 2.1 - Implementada lógica para mano de 5 dedos
# =================================================================

import mediapipe as mp
import numpy as np
import math
import time
import cv2
import config

# --- FUNCIÓN MAESTRA PARA OBTENER TODOS LOS ÁNGULOS ---
def get_all_raw_angles(pose_results, hand_results, h, w, calibracion_completada, dist_ref, dist_rot_ref):
    """
    Calcula todos los ángulos brutos para los 10 ejes del robot basándose en
    los resultados de detección de pose y manos.
    """
    raw_angles = {'proximidad': 90, 'hombro': 90, 'codo': 90, 'pitch': 90, 'roll': 90}
    muneca = [0, 0]

    if pose_results and pose_results.pose_landmarks:
        ang_brazo, codo, muneca = _calcular_angulos_brazo(pose_results.pose_landmarks, h, w)
        raw_angles.update(ang_brazo)

    if hand_results and hand_results.multi_hand_landmarks:
        hand_lm = hand_results.multi_hand_landmarks[0]
        
        gestos_mano = _calcular_gestos_mano(hand_lm, dist_rot_ref)
        raw_angles.update(gestos_mano)

        if calibracion_completada:
            distancia_mano_actual = _calcular_distancia_mano(hand_lm)
            raw_angles['proximidad'] = _calcular_proximidad_bruta(distancia_mano_actual, dist_ref)

    return raw_angles, muneca

# --- CLASE PARA SUAVIZADO Y PRUEBAS ---
class AngleProcessor:
    """Gestiona el estado de los ángulos, su suavizado y los modos de prueba."""
    def __init__(self):
        self.smoothed_angles = {
            'proximidad': 90.0, 'hombro': 90.0, 'codo': 90.0,
            'pitch': 90.0, 'roll': 90.0, 
            'pulgar': 90.0, 'indice': 90.0, 'medio': 90.0,
            'anular': 90.0, 'menique': 90.0
        }

    def get_current_angles(self):
        """Devuelve una copia de los ángulos actuales, ideal para iniciar el modo manual."""
        return {key: int(value) for key, value in self.smoothed_angles.items()}

    def smooth_angles(self, raw_angles, hand_results=None, test_servo_key=None, modo_actual=config.MODO_NORMAL):
        target_angles = {}

        if modo_actual == config.MODO_MANUAL:
            self.smoothed_angles.update(raw_angles)
            return raw_angles
        
        elif modo_actual == config.MODO_PRUEBA:
            target_angles = {}
            if test_servo_key == 'all':
                speed = config.TEST_ALL_SERVOS_SPEED
                # Lógica de prueba para todos los servos (sin la antigua 'mano')
                target_angles['proximidad'] = np.interp(math.sin(time.time() * speed), [-1, 1], [45, 135])
                target_angles['hombro'] = np.interp(math.sin(time.time() * speed + 1), [-1, 1], [60, 150])
                target_angles['codo'] = np.interp(math.sin(time.time() * speed + 2), [-1, 1], [45, 135])
                target_angles['pitch'] = np.interp(math.sin(time.time() * speed + 3), [-1, 1], [45, 135])
                target_angles['roll'] = np.interp(math.sin(time.time() * speed + 4), [-1, 1], [45, 135])
                # Podrías añadir lógica para los dedos si quisieras
            else:
                sweep_angle = np.interp(math.sin(time.time() * config.TEST_SWEEP_SPEED), [-1, 1], [config.TEST_SWEEP_MIN, config.TEST_SWEEP_MAX])
                for key in self.smoothed_angles.keys():
                    target_angles[key] = sweep_angle if key == test_servo_key else 90
        
        elif modo_actual == config.MODO_GESTO_SI:
            target_angles = self.smoothed_angles.copy()
            target_angles['codo'] = 135
            target_angles['pitch'] = np.interp(math.sin(time.time() * 6), [-1, 1], [45, 135])
            # Actualizamos los dedos con los valores actuales de la visión
            if hand_results and hand_results.multi_hand_landmarks:
                dedos_actuales = _calcular_angulos_dedos(hand_results.multi_hand_landmarks[0])
                target_angles.update(dedos_actuales)

        elif modo_actual == config.MODO_GESTO_NO:
            target_angles = self.smoothed_angles.copy()
            target_angles['roll'] = np.interp(math.sin(time.time() * 6), [-1, 1], [45, 135])
            # Actualizamos los dedos con los valores actuales de la visión
            if hand_results and hand_results.multi_hand_landmarks:
                dedos_actuales = _calcular_angulos_dedos(hand_results.multi_hand_landmarks[0])
                target_angles.update(dedos_actuales)

        else:
            target_angles = raw_angles
        
        smoothed_output = {}
        for key, raw_value in target_angles.items():
            if key in self.smoothed_angles and raw_value is not None:
                factor = config.PROXIMITY_FILTER_FACTOR if key == 'proximidad' else config.ANGLE_SMOOTHING_FACTOR
                self.smoothed_angles[key] = (factor * raw_value) + ((1 - factor) * self.smoothed_angles[key])
                smoothed_output[key] = int(self.smoothed_angles[key])
            else:
                smoothed_output[key] = raw_value
        return smoothed_output

# --- CLASE POSE DETECTOR ---
class PoseDetector:
    """Encapsula la funcionalidad de MediaPipe para la detección de pose y manos."""
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.pose = self.mp_pose.Pose(model_complexity=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.hands = self.mp_hands.Hands(model_complexity=1, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

    def find_pose(self, image): 
        return self.pose.process(image)
        
    def find_hands(self, image): 
        return self.hands.process(image)
        
    def draw_all_landmarks(self, image, pose_results, hand_results):
        if pose_results and pose_results.pose_landmarks: 
            self.mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS, 
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=5), 
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(180, 180, 180), thickness=3))
        if hand_results and hand_results.multi_hand_landmarks:
            for hand_lm in hand_results.multi_hand_landmarks: 
                self.mp_drawing.draw_landmarks(image, hand_lm, self.mp_hands.HAND_CONNECTIONS, 
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=5), 
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(180, 180, 180), thickness=2))

# --- FUNCIONES AUXILIARES (PRIVADAS) ---
def _calcular_proximidad_bruta(distancia_actual, distancia_referencia):
    if distancia_referencia > 0:
        dist_min = distancia_referencia - (distancia_referencia * config.DISTANCE_RANGE)
        dist_max = distancia_referencia + (distancia_referencia * config.DISTANCE_RANGE)
        return np.interp(distancia_actual, [dist_min, dist_max], [0, 180])
    return 90

def _calcular_angulos_brazo(landmarks, h, w):
    lm = landmarks.landmark
    shoulder = [lm[12].x * w, lm[12].y * h]
    elbow = [lm[14].x * w, lm[14].y * h]
    wrist = [lm[16].x * w, lm[16].y * h]
    
    vec_shoulder_elbow = [elbow[0] - shoulder[0], elbow[1] - shoulder[1]]
    mag_vec = np.linalg.norm(vec_shoulder_elbow)
    ang_hombro = 90
    if mag_vec > 0: 
        ang_hombro = np.degrees(np.arccos(max(min(np.dot(vec_shoulder_elbow, [0, -1]) / mag_vec, 1), -1)))
        
    vec1 = [shoulder[0] - elbow[0], shoulder[1] - elbow[1]]
    vec2 = [wrist[0] - elbow[0], wrist[1] - elbow[1]]
    mag1, mag2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    ang_codo = 90
    if mag1 > 0 and mag2 > 0: 
        ang_codo = np.degrees(np.arccos(max(min(np.dot(vec1, vec2) / (mag1 * mag2), 1), -1)))
        
    return ({'hombro': ang_hombro, 'codo': ang_codo}, elbow, wrist)

def _calcular_distancia_mano(hand_landmarks, punto1=5, punto2=17):
    punto_a = hand_landmarks.landmark[punto1]
    punto_b = hand_landmarks.landmark[punto2]
    return math.sqrt((punto_a.x - punto_b.x)**2 + (punto_a.y - punto_b.y)**2)

def _calcular_angulos_dedos(hand_landmarks):
    """Calcula el ángulo de cada dedo basándose en la distancia de la punta a la muñeca."""
    dedos = {}
    
    puntas_dedos_ids = {
        'pulgar': 4, 'indice': 8, 'medio': 12, 'anular': 16, 'menique': 20
    }
    
    punto_base = hand_landmarks.landmark[0]

    # --- ¡IMPORTANTE! DEBES AJUSTAR ESTOS VALORES PARA TU MANO ---
    rangos_distancia = {
        'pulgar':  {'min': 0.05, 'max': 0.15, 'ang_min': 0, 'ang_max': 160},
        'indice':  {'min': 0.08, 'max': 0.25, 'ang_min': 0, 'ang_max': 180},
        'medio':   {'min': 0.08, 'max': 0.27, 'ang_min': 0, 'ang_max': 180},
        'anular':  {'min': 0.08, 'max': 0.25, 'ang_min': 0, 'ang_max': 180},
        'menique': {'min': 0.08, 'max': 0.22, 'ang_min': 0, 'ang_max': 180},
    }

    for dedo, punta_id in puntas_dedos_ids.items():
        punto_punta = hand_landmarks.landmark[punta_id]
        distancia = math.sqrt((punto_punta.x - punto_base.x)**2 + (punto_punta.y - punto_base.y)**2)
        
        rango = rangos_distancia[dedo]
        angulo = np.interp(distancia, [rango['min'], rango['max']], [rango['ang_max'], rango['ang_min']])
        
        dedos[dedo] = max(0, min(180, int(angulo)))
        
    return dedos

def _calcular_gestos_mano(hand_landmarks, distancia_rotacion_ref):
    """Calcula los ángulos de pitch, roll y los 5 dedos."""
    ma = 90
    try:
        wrist_y = hand_landmarks.landmark[0].y
        mcp_y = hand_landmarks.landmark[9].y
        y_diff = wrist_y - mcp_y
        ma = np.interp(y_diff, [config.PITCH_INPUT_RANGE_MIN, config.PITCH_INPUT_RANGE_MAX], [0, 180])
    except: pass

    mr = 90
    try:
        distancia_actual_rotacion = _calcular_distancia_mano(hand_landmarks, 5, 0)
        if distancia_rotacion_ref > 0:
            variacion_rotacion = distancia_actual_rotacion - distancia_rotacion_ref
            mr = np.interp(variacion_rotacion, [-config.ROLL_INPUT_RANGE, config.ROLL_INPUT_RANGE], [config.ROLL_OUTPUT_MIN_ANGLE, config.ROLL_OUTPUT_MAX_ANGLE])
        else:
            mr_raw = np.interp(hand_landmarks.landmark[5].x - hand_landmarks.landmark[17].x, [-config.ROLL_INPUT_RANGE, config.ROLL_INPUT_RANGE], [180, 0])
            mr = np.interp(mr_raw, [0, 180], [config.ROLL_OUTPUT_MIN_ANGLE, config.ROLL_OUTPUT_MAX_ANGLE])
    except: pass
    
    angulos_dedos = _calcular_angulos_dedos(hand_landmarks)
    
    resultado_final = {'pitch': ma, 'roll': mr}
    resultado_final.update(angulos_dedos)
    
    return resultado_final

def aplicar_limites_seguros(angulos):
    """Aplica los límites de seguridad definidos en config.py a cada ángulo."""
    angulos_limitados = {}
    for eje, valor in angulos.items():
        if eje in config.ANGULOS_SEGUROS:
            min_val, max_val = config.ANGULOS_SEGUROS[eje]
            angulos_limitados[eje] = max(min_val, min(max_val, int(valor)))
        else:
            angulos_limitados[eje] = int(valor)
    return angulos_limitados