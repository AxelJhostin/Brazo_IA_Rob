# =================================================================
# PROYECTO: Control de Brazo Robótico con Visión (6 Ejes)
# VERSIÓN: 1.8.1 - Corregida la detección de teclas de flecha
# =================================================================

import cv2
import numpy as np
import time
from collections import deque
import os
import socket

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
    
    # --- Variables para el Modo Manual ---
    articulacion_seleccionada_idx = 0
    manual_angles = {}
    articulaciones_keys = ['proximidad', 'hombro', 'codo', 'pitch', 'roll', 'mano']
    
    angle_processor = robot_logic.AngleProcessor()

    cv2.namedWindow("Control de Brazo Robotico", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Control de Brazo Robotico", 1600, 900)
    logo_img = cv2.imread("logo_puce.png") if os.path.exists("logo_puce.png") else None
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        conexion_brazo = True
        print(f"Socket creado. Enviando datos a {config.ESP_IP}:{config.ESP_PORT}")
    except Exception as e:
        sock = None; conexion_brazo = False; print(f"Error al crear el socket: {e}")

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened(): print("Error: No se pudo abrir la cámara"); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    
    detector = robot_logic.PoseDetector()
    gesture_buffer = deque(maxlen=config.GESTURE_BUFFER_SIZE)

    key_actions = {
        ord(' '): (config.MODO_CONFIGURACION, None, "Modo cambiado a: CONFIGURACION"),
        ord('o'): (config.MODO_NORMAL, None, "Modo cambiado a: NORMAL"),
        ord('a'): (config.MODO_POSTURA, 'saludo', "Activando postura: 'saludo'"),
        ord('z'): (config.MODO_POSTURA, 'home', "Activando postura: 'home'"),
        ord('s'): (config.MODO_GESTO_SI, None, "Activando Gesto: 'SI'"),
        ord('n'): (config.MODO_GESTO_NO, None, "Activando Gesto: 'NO'"),
    }
    
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
        
        if modo_actual == config.MODO_MANUAL:
            final_raw_angles = manual_angles
        elif modo_actual == config.MODO_PRUEBA:
            test_key = servo_en_prueba
        elif modo_actual == config.MODO_POSTURA and postura_activa:
            final_raw_angles = config.POSTURAS_PREDEFINIDAS[postura_activa]
        elif modo_actual in [config.MODO_GESTO_SI, config.MODO_GESTO_NO]:
            if results_hands and results_hands.multi_hand_landmarks:
                final_raw_angles['pinza'] = robot_logic._detectar_pinza(results_hands.multi_hand_landmarks[0])
        else:
            final_raw_angles, muneca = robot_logic.get_all_raw_angles(
                results_pose, results_hands, h, w, calibracion_completada,
                distancia_referencia, distancia_rotacion_referencia
            )
        
        if modo_actual not in [config.MODO_PRUEBA, config.MODO_MANUAL]:
             if results_hands and results_hands.multi_hand_landmarks:
                gesture_buffer.append(robot_logic._detectar_pinza(results_hands.multi_hand_landmarks[0]))
        
        mano_estable = 1 if sum(gesture_buffer) >= config.GESTURE_CONFIRMATION_THRESHOLD else 0
        if 'mano' not in final_raw_angles:
            final_raw_angles['mano'] = mano_estable

        angulos_finales = angle_processor.smooth_angles(final_raw_angles, test_servo_key=test_key, modo_actual=modo_actual)
        
        if modo_actual == config.MODO_MANUAL:
            manual_angles = angulos_finales.copy()

        if conexion_brazo and modo_actual != config.MODO_PAUSA:
            angulos_seguros = robot_logic.aplicar_limites_seguros(angulos_finales)
            datos = f"<{int(angulos_seguros['proximidad'])},{int(angulos_seguros['hombro'])},{int(angulos_seguros['codo'])},{int(angulos_seguros['pitch'])},{int(angulos_seguros['roll'])},{int(angulos_seguros['mano'])}>"
            sock.sendto(datos.encode('utf-8'), (config.ESP_IP, config.ESP_PORT))

        lienzo = np.zeros((h + 100, w + 450, 3), dtype=np.uint8)
        panel_sup = crear_panel_superior(w + 450, 100, logo_img)
        lienzo[0:100, 0:w+450] = panel_sup
        detector.draw_all_landmarks(frame, results_pose, results_hands)
        lienzo[100:100+h, 0:w] = frame
        
        panel_lat = crear_panel_lateral(450, h, angulos_finales, mano_estable, conexion_brazo, modo_actual, servo_en_prueba, 0, postura_activa, articulacion_seleccionada_idx)
        lienzo[100:100+h, w:w+450] = panel_lat
        cv2.imshow("Control de Brazo Robotico", lienzo)
        
        # ¡CORRECCIÓN! Usamos waitKeyEx para capturar teclas especiales como las flechas.
        key = cv2.waitKeyEx(5)

        if key != -1:
            if key == 27: break

            if modo_actual == config.MODO_MANUAL:
                key_str = articulaciones_keys[articulacion_seleccionada_idx]
                if key == 2490368: # Flecha Arriba
                    articulacion_seleccionada_idx = (articulacion_seleccionada_idx - 1) % len(articulaciones_keys)
                elif key == 2621440: # Flecha Abajo
                    articulacion_seleccionada_idx = (articulacion_seleccionada_idx + 1) % len(articulaciones_keys)
                elif key == 2424832: # Flecha Izquierda
                    if key_str == 'mano': manual_angles[key_str] = 0
                    else: manual_angles[key_str] -= 2
                elif key == 2555904: # Flecha Derecha
                    if key_str == 'mano': manual_angles[key_str] = 1
                    else: manual_angles[key_str] += 2
            
            if key in key_actions:
                modo_actual, postura_activa, msg = key_actions[key]
                servo_en_prueba = None
                print(msg)
            elif key == ord('p'):
                modo_actual = config.MODO_PAUSA; print("Modo cambiado a: PAUSA")
            elif key == ord('m'):
                modo_actual = config.MODO_MANUAL
                manual_angles = angle_processor.get_current_angles()
                print("Modo cambiado a: MANUAL")
            elif ord('1') <= key <= ord('7'):
                servo_map = {'1':'proximidad','2':'hombro','3':'codo','4':'pitch','5':'roll','6':'mano', '7':'all'}
                servo_en_prueba = servo_map.get(chr(key))
                if servo_en_prueba:
                    modo_actual, postura_activa = config.MODO_PRUEBA, None
                    print(f"Modo cambiado a: PRUEBA - Servo: {servo_en_prueba}")

    cap.release()
    if sock: sock.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
