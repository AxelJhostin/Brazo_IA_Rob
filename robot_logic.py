# =================================================================
# MÓDULO: robot_logic.py
# DESCRIPCIÓN: Contiene toda la lógica de negocio para el control
#              del brazo: detección de pose, cálculo de ángulos y gestos.
# VERSIÓN: 1.6 - Corregido Movimiento de Muñeca (Pitch) (26/07/2024)
# =================================================================

import mediapipe as mp
import numpy as np
import math
import time
import cv2
import config

# --- FUNCIÓN MAESTRA PARA OBTENER TODOS LOS ÁNGULOS ---
def get_all_raw_angles(pose_results, hand_results, h, w, calibracion_completada, dist_ref, dist_rot_ref):
    raw_angles = {'proximidad': 90, 'hombro': 90, 'codo': 90, 'pitch': 90, 'roll': 90, 'pinza': 0}
    muneca = [0, 0]

    if pose_results and pose_results.pose_landmarks:
        ang_brazo, codo, muneca = _calcular_angulos_brazo(pose_results.pose_landmarks, h, w)
        raw_angles.update(ang_brazo)

        if hand_results and hand_results.multi_hand_landmarks:
            hand_lm = hand_results.multi_hand_landmarks[0]
            gestos = _calcular_gestos_mano(hand_lm, dist_rot_ref)
            raw_angles.update({'pitch': gestos['pitch'], 'roll': gestos['roll_raw'], 'pinza': gestos['pinza']})

            if calibracion_completada:
                distancia_mano_actual = _calcular_distancia_mano(hand_lm)
                raw_angles['proximidad'] = _calcular_proximidad_bruta(distancia_mano_actual, dist_ref)

    return raw_angles, muneca

# --- CLASE PARA SUAVIZADO Y PRUEBAS ---
class AngleProcessor:
    def __init__(self):
        self.smoothed_angles = {
            'proximidad': 90.0, 'hombro': 90.0, 'codo': 90.0,
            'pitch': 90.0, 'roll': 90.0
        }

    def smooth_angles(self, raw_angles, test_servo_key=None):
        target_angles = raw_angles
        if test_servo_key:
            target_angles = {}
            if test_servo_key == 'all':
                speed = config.TEST_ALL_SERVOS_SPEED
                target_angles['proximidad'] = np.interp(math.sin(time.time() * speed), [-1, 1], [45, 135])
                target_angles['hombro'] = np.interp(math.sin(time.time() * speed + 1), [-1, 1], [60, 150])
                target_angles['codo'] = np.interp(math.sin(time.time() * speed + 2), [-1, 1], [45, 135])
                target_angles['pitch'] = np.interp(math.sin(time.time() * speed + 3), [-1, 1], [45, 135])
                target_angles['roll'] = np.interp(math.sin(time.time() * speed + 4), [-1, 1], [45, 135])
                target_angles['mano'] = 1 if math.sin(time.time() * speed * 0.5) > 0 else 0
            else:
                sweep_angle = np.interp(math.sin(time.time() * config.TEST_SWEEP_SPEED), [-1, 1], [config.TEST_SWEEP_MIN, config.TEST_SWEEP_MAX])
                for key in self.smoothed_angles.keys():
                    target_angles[key] = sweep_angle if key == test_servo_key else 90
                target_angles['mano'] = 1 if sweep_angle > 90 else 0 if test_servo_key == 'mano' else raw_angles.get('mano', 0)
        
        smoothed_output = {}
        for key, raw_value in target_angles.items():
            if key in self.smoothed_angles:
                factor = config.PROXIMITY_FILTER_FACTOR if key == 'proximidad' else config.ANGLE_SMOOTHING_FACTOR
                self.smoothed_angles[key] = (factor * raw_value) + ((1 - factor) * self.smoothed_angles[key])
                smoothed_output[key] = int(self.smoothed_angles[key])
            else:
                smoothed_output[key] = raw_value
        return smoothed_output

# --- CLASE POSE DETECTOR ---
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
        if pose_results and pose_results.pose_landmarks: 
            self.mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=5), connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(180, 180, 180), thickness=3))
            h, w, _ = image.shape
            for idx in [11, 12]:
                if idx < len(pose_results.pose_landmarks.landmark):
                    lm = pose_results.pose_landmarks.landmark[idx]; cx, cy = int(lm.x * w), int(lm.y * h); cv2.circle(image, (cx, cy), 12, config.COLORES['hombro'],