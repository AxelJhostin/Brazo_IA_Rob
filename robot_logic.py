# =================================================================
# MÓDULO: robot_logic.py
# VERSIÓN: 2.2 - CORRECCIONES: Proximidad sin calibración obligatoria
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
    raw_angles = {'proximidad': 90, 'hombro': 90, 'codo': 90, 'pitch': 90, 'roll': 90,
                  'pulgar': 90, 'indice': 90, 'medio': 90, 'anular': 90, 'menique': 90}
    muneca = [0, 0]

    # Calcular ángulos del brazo desde la pose
    if pose_results and pose_results.pose_landmarks:
        ang_brazo, codo, muneca = _calcular_angulos_brazo(pose_results.pose_landmarks, h, w)
        raw_angles.update(ang_brazo)

    # Calcular gestos de la mano y dedos
    if hand_results and hand_results.multi_hand_landmarks:
        hand_lm = hand_results.multi_hand_landmarks[0]
        
        # Calcular pitch, roll y los 5 dedos
        gestos_mano = _calcular_gestos_mano(hand_lm, dist_rot_ref)
        raw_angles.update(gestos_mano)

        # CORRECCIÓN: Permitir proximidad incluso sin calibración
        distancia_mano_actual = _calcular_distancia_mano(hand_lm)
        
        if calibracion_completada and dist_ref > 0:
            # Con calibración: usar la distancia de referencia
            raw_angles['proximidad'] = _calcular_proximidad_bruta(distancia_mano_actual, dist_ref)
        else:
            # Sin calibración: mapear la distancia a un rango fijo
            # Ajusta estos valores según tu distancia típica de la mano a la cámara
            raw_angles['proximidad'] = np.interp(distancia_mano_actual, [0.08, 0.35], [0, 180])

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

        # Modo Manual: usar directamente los ángulos sin suavizado
        if modo_actual == config.MODO_MANUAL:
            self.smoothed_angles.update(raw_angles)
            return raw_angles
        
        # Modo Prueba: generar patrones de prueba
        elif modo_actual == config.MODO_PRUEBA:
            target_angles = {}
            if test_servo_key == 'all':
                speed = config.TEST_ALL_SERVOS_SPEED
                # Prueba para todos los servos con desfase
                target_angles['proximidad'] = np.interp(math.sin(time.time() * speed), [-1, 1], [45, 135])
                target_angles['hombro'] = np.interp(math.sin(time.time() * speed + 1), [-1, 1], [60, 150])
                target_angles['codo'] = np.interp(math.sin(time.time() * speed + 2), [-1, 1], [45, 135])
                target_angles['pitch'] = np.interp(math.sin(time.time() * speed + 3), [-1, 1], [45, 135])
                target_angles['roll'] = np.interp(math.sin(time.time() * speed + 4), [-1, 1], [45, 135])
                target_angles['pulgar'] = np.interp(math.sin(time.time() * speed + 5), [-1, 1], [20, 160])
                target_angles['indice'] = np.interp(math.sin(time.time() * speed + 6), [-1, 1], [10, 150])
                target_angles['medio'] = np.interp(math.sin(time.time() * speed + 7), [-1, 1], [10, 150])
                target_angles['anular'] = np.interp(math.sin(time.time() * speed + 8), [-1, 1], [10, 150])
                target_angles['menique'] = np.interp(math.sin(time.time() * speed + 9), [-1, 1], [10, 150])
            else:
                # Prueba individual: un servo se mueve, el resto en 90
                sweep_angle = np.interp(math.sin(time.time() * config.TEST_SWEEP_SPEED), 
                                       [-1, 1], 
                                       [config.TEST_SWEEP_MIN, config.TEST_SWEEP_MAX])
                for key in self.smoothed_angles.keys():
                    target_angles[key] = sweep_angle if key == test_servo_key else 90
        
        # Modo Gesto SI: codo arriba, pitch oscilando
        elif modo_actual == config.MODO_GESTO_SI:
            target_angles = self.smoothed_angles.copy()
            target_angles['codo'] = 135
            target_angles['pitch'] = np.interp(math.sin(time.time() * 6), [-1, 1], [45, 135])
            # Mantener control de dedos con la visión
            if hand_results and hand_results.multi_hand_landmarks:
                dedos_actuales = _calcular_angulos_dedos(hand_results.multi_hand_landmarks[0])
                target_angles.update(dedos_actuales)

        # Modo Gesto NO: roll oscilando
        elif modo_actual == config.MODO_GESTO_NO:
            target_angles = self.smoothed_angles.copy()
            target_angles['roll'] = np.interp(math.sin(time.time() * 6), [-1, 1], [45, 135])
            # Mantener control de dedos con la visión
            if hand_results and hand_results.multi_hand_landmarks:
                dedos_actuales = _calcular_angulos_dedos(hand_results.multi_hand_landmarks[0])
                target_angles.update(dedos_actuales)

        # Modo Normal u otros: usar los ángulos calculados
        else:
            target_angles = raw_angles
        
        # Aplicar suavizado
        smoothed_output = {}
        for key, raw_value in target_angles.items():
            if key in self.smoothed_angles and raw_value is not None:
                # Factor de suavizado diferente para proximidad
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
        
        # Configuración optimizada para detección
        self.pose = self.mp_pose.Pose(
            model_complexity=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        self.hands = self.mp_hands.Hands(
            model_complexity=1,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def find_pose(self, image): 
        return self.pose.process(image)
        
    def find_hands(self, image): 
        return self.hands.process(image)
        
    def draw_all_landmarks(self, image, pose_results, hand_results):
        """Dibuja todos los landmarks detectados en la imagen."""
        # Dibujar pose (esqueleto del cuerpo)
        if pose_results and pose_results.pose_landmarks: 
            self.mp_drawing.draw_landmarks(
                image, 
                pose_results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    color=(0, 0, 255), thickness=3, circle_radius=5
                ),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    color=(180, 180, 180), thickness=3
                )
            )
        
        # Dibujar mano
        if hand_results and hand_results.multi_hand_landmarks:
            for hand_lm in hand_results.multi_hand_landmarks: 
                self.mp_drawing.draw_landmarks(
                    image, 
                    hand_lm, 
                    self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                        color=(0, 255, 0), thickness=3, circle_radius=5
                    ),
                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                        color=(180, 180, 180), thickness=2
                    )
                )

# --- FUNCIONES AUXILIARES (PRIVADAS) ---

def _calcular_angulo_3p(p1, p2, p3):
    """
    Calcula el ángulo (en grados) en el vértice p2, formado por p1-p2-p3.
    Usa coordenadas 2D para mayor estabilidad.
    """
    # Obtener coordenadas 2D
    a = np.array([p1.x, p1.y])
    b = np.array([p2.x, p2.y])  # Vértice
    c = np.array([p3.x, p3.y])

    # Calcular vectores
    vec1 = a - b
    vec2 = c - b

    # Calcular el producto punto y las magnitudes
    dot_prod = np.dot(vec1, vec2)
    mag1 = np.linalg.norm(vec1)
    mag2 = np.linalg.norm(vec2)

    if mag1 == 0 or mag2 == 0:
        return 180.0  # Asumir recto si no hay vector

    # Calcular el coseno del ángulo (con clip para evitar errores numéricos)
    cosine_angle = np.clip(dot_prod / (mag1 * mag2), -1.0, 1.0)

    # Calcular el ángulo en radianes y convertir a grados
    angle_rad = np.arccos(cosine_angle)
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def _calcular_proximidad_bruta(distancia_actual, distancia_referencia):
    """Calcula el ángulo de proximidad basado en la distancia de la mano."""
    if distancia_referencia > 0:
        dist_min = distancia_referencia - (distancia_referencia * config.DISTANCE_RANGE)
        dist_max = distancia_referencia + (distancia_referencia * config.DISTANCE_RANGE)
        return np.interp(distancia_actual, [dist_min, dist_max], [0, 180])
    return 90

def _calcular_angulos_brazo(landmarks, h, w):
    """Calcula los ángulos del hombro y codo desde los landmarks de pose."""
    lm = landmarks.landmark
    
    # Obtener puntos clave (brazo derecho)
    shoulder = [lm[12].x * w, lm[12].y * h]
    elbow = [lm[14].x * w, lm[14].y * h]
    wrist = [lm[16].x * w, lm[16].y * h]
    
    # Calcular ángulo del hombro (respecto a la vertical)
    vec_shoulder_elbow = [elbow[0] - shoulder[0], elbow[1] - shoulder[1]]
    mag_vec = np.linalg.norm(vec_shoulder_elbow)
    ang_hombro = 90
    
    if mag_vec > 0: 
        # Ángulo con respecto al eje vertical (hacia abajo = [0, -1])
        dot_product = np.dot(vec_shoulder_elbow, [0, -1])
        cos_angle = np.clip(dot_product / mag_vec, -1.0, 1.0)
        ang_hombro = 180 - np.degrees(np.arccos(cos_angle))
    
    # Calcular ángulo del codo (entre hombro-codo-muñeca)
    vec1 = [shoulder[0] - elbow[0], shoulder[1] - elbow[1]]
    vec2 = [wrist[0] - elbow[0], wrist[1] - elbow[1]]
    mag1, mag2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    ang_codo = 90
    
    if mag1 > 0 and mag2 > 0: 
        dot_product = np.dot(vec1, vec2)
        cos_angle = np.clip(dot_product / (mag1 * mag2), -1.0, 1.0)
        ang_codo = 180 - np.degrees(np.arccos(cos_angle))
        
    return ({'hombro': ang_hombro, 'codo': ang_codo}, elbow, wrist)

def _calcular_distancia_mano(hand_landmarks, punto1=5, punto2=17):
    """Calcula la distancia euclidiana entre dos puntos de la mano."""
    punto_a = hand_landmarks.landmark[punto1]
    punto_b = hand_landmarks.landmark[punto2]
    return math.sqrt((punto_a.x - punto_b.x)**2 + (punto_a.y - punto_b.y)**2)

def _calcular_angulos_dedos(hand_landmarks):
    """
    Calcula el ángulo de cada dedo basándose en el ángulo interno
    de las articulaciones (MCP-PIP-TIP).
    """
    dedos = {}
    lm = hand_landmarks.landmark

    # Rangos de mapeo (ajusta según tu robot)
    # [Ángulo humano doblado, recto] -> [Servo doblado, recto]
    INPUT_ANGLE_MIN = 80.0   # Dedo doblado
    INPUT_ANGLE_MAX = 170.0  # Dedo recto
    
    THUMB_INPUT_MIN = 130.0  # Pulgar tiene rango diferente
    THUMB_INPUT_MAX = 170.0
    
    SERVO_ANGLE_MIN = 0      # Servo cerrado
    SERVO_ANGLE_MAX = 180    # Servo abierto

    # --- Dedo Índice ---
    try:
        angulo_indice = _calcular_angulo_3p(lm[5], lm[6], lm[8])
        dedos['indice'] = np.interp(angulo_indice, [INPUT_ANGLE_MIN, INPUT_ANGLE_MAX], 
                                     [SERVO_ANGLE_MIN, SERVO_ANGLE_MAX])
    except Exception:
        dedos['indice'] = 90

    # --- Dedo Medio ---
    try:
        angulo_medio = _calcular_angulo_3p(lm[9], lm[10], lm[12])
        dedos['medio'] = np.interp(angulo_medio, [INPUT_ANGLE_MIN, INPUT_ANGLE_MAX], 
                                    [SERVO_ANGLE_MIN, SERVO_ANGLE_MAX])
    except Exception:
        dedos['medio'] = 90

    # --- Dedo Anular ---
    try:
        angulo_anular = _calcular_angulo_3p(lm[13], lm[14], lm[16])
        dedos['anular'] = np.interp(angulo_anular, [INPUT_ANGLE_MIN, INPUT_ANGLE_MAX], 
                                    [SERVO_ANGLE_MAX, SERVO_ANGLE_MIN])
    except Exception:
        dedos['anular'] = 90

    # --- Dedo Meñique ---
    try:
        angulo_menique = _calcular_angulo_3p(lm[17], lm[18], lm[20])
        dedos['menique'] = np.interp(angulo_menique, [INPUT_ANGLE_MIN, INPUT_ANGLE_MAX], 
                                      [SERVO_ANGLE_MIN, SERVO_ANGLE_MAX])
    except Exception:
        dedos['menique'] = 90

    # --- Pulgar ---
    try:
        angulo_pulgar = _calcular_angulo_3p(lm[2], lm[3], lm[4])
        dedos['pulgar'] = np.interp(angulo_pulgar, [THUMB_INPUT_MIN, THUMB_INPUT_MAX], 
                                     [SERVO_ANGLE_MIN, SERVO_ANGLE_MAX])
    except Exception:
        dedos['pulgar'] = 90

    # Asegurar que todos los valores estén en el rango 0-180
    for key in dedos:
        dedos[key] = max(0, min(180, int(dedos[key])))
        
    return dedos

def _calcular_gestos_mano(hand_landmarks, distancia_rotacion_ref):
    """Calcula los ángulos de pitch, roll y los 5 dedos."""
    
    # --- Calcular Pitch (inclinación de la mano) ---
    ma = 90
    try:
        wrist_y = hand_landmarks.landmark[0].y
        mcp_y = hand_landmarks.landmark[9].y
        y_diff = wrist_y - mcp_y
        ma = np.interp(y_diff, 
                      [config.PITCH_INPUT_RANGE_MIN, config.PITCH_INPUT_RANGE_MAX], 
                      [0, 180])
    except Exception:
        pass

    # --- Calcular Roll (rotación de la muñeca) ---
    mr = 90
    try:
        distancia_actual_rotacion = _calcular_distancia_mano(hand_landmarks, 5, 0)
        
        if distancia_rotacion_ref > 0:
            # Con referencia de calibración
            variacion_rotacion = distancia_actual_rotacion - distancia_rotacion_ref
            mr = np.interp(variacion_rotacion, 
                          [-config.ROLL_INPUT_RANGE, config.ROLL_INPUT_RANGE], 
                          [config.ROLL_OUTPUT_MIN_ANGLE, config.ROLL_OUTPUT_MAX_ANGLE])
        else:
            # Sin calibración: usar diferencia x directa
            mr_raw = np.interp(hand_landmarks.landmark[5].x - hand_landmarks.landmark[17].x, 
                              [-config.ROLL_INPUT_RANGE, config.ROLL_INPUT_RANGE], 
                              [180, 0])
            mr = np.interp(mr_raw, [0, 180], 
                          [config.ROLL_OUTPUT_MIN_ANGLE, config.ROLL_OUTPUT_MAX_ANGLE])
    except Exception:
        pass
    
    # --- Calcular ángulos de los 5 dedos ---
    angulos_dedos = _calcular_angulos_dedos(hand_landmarks)
    
    # Combinar todo
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
            # Si no hay límite definido, usar 0-180 por defecto
            angulos_limitados[eje] = max(0, min(180, int(valor)))
    
    return angulos_limitados