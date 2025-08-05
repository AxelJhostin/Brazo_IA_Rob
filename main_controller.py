# =================================================================
# PROYECTO: Control de Brazo Robótico con Visión (6 Ejes)
# VERSIÓN: 1.4.0 - Adaptado para Wi-Fi (UDP)
# =================================================================

import cv2
import numpy as np
import time
from collections import deque
import os
import socket  # AÑADIDO: Para comunicación por red

import config
import robot_logic
from ui_components import crear_panel_superior, crear_panel_lateral, dibujar_zona_calibracion

def main():
    # --- INICIALIZACIÓN ---
    modo_actual = config.MODO_NORMAL
    postura_activa, servo_en_prueba = None, None
    calibracion_completada = False
    distancia_referencia, distancia_rotacion_referencia = 0, 0
    tiempo_inicio_calibracion = 0
    
    angle_processor = robot_logic.AngleProcessor()

    cv2.namedWindow("Control de Brazo Robotico", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Control de Brazo Robotico", 1600, 900)
    logo_img = cv2.imread("logo_puce.png") if os.path.exists("logo_puce.png") else None
    
    # --- CONFIGURACIÓN DE RED UDP ---
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        conexion_brazo = True
        print(f"Socket creado. Enviando datos a {config.ESP_IP}:{config.ESP_PORT}")
    except Exception as e:
        sock = None
        conexion_brazo = False
        print(f"Error al crear el socket: {e}")

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened(): print("Error: No se pudo abrir la cámara"); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    
    detector = robot_logic.PoseDetector()
    gesture_buffer = deque(maxlen=config.GESTURE_BUFFER_SIZE)
    
    # --- BUCLE PRINCIPAL ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results_pose = detector.find_pose(image_rgb)
        results_hands = detector.find_hands(image_rgb)

        final_raw_angles = {}
        test_key = None
        
        if modo_actual == config.MODO_PRUEBA:
            test_key = servo_