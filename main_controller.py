# =================================================================
# PROYECTO: Control de Brazo Robótico con Visión (6 Ejes)
# VERSIÓN: 1.3.8 - Corregido Bloqueo de Calibración (26/07/2024)
# =================================================================

import cv2
import numpy as np
import time
from collections import deque
import os
import serial
import serial.tools.list_ports

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
    
    try:
        arduino = serial.Serial(config.SERIAL_PORT, config.BAUD_RATE, timeout=0.1)
        time.sleep(1); print(f"Serial conectado en {config.SERIAL_PORT}")
    except Exception as e:
        arduino = None; print(f"Error de conexión serial: {e}")

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

        # --- LÓGICA DE MODOS REESTRUCTURADA ---
        final_raw_angles = {}
        test_key = None
        
        if modo_actual == config.MODO_PRUEBA:
            test_key = servo_en_prueba
            # Pasamos un diccionario vacío porque el AngleProcessor generará los valores
            final_raw_angles = {} 
        elif modo_actual == config.MODO_POSTURA and postura_activa:
            final_raw_angles = config.POSTURAS_PREDEFINIDAS[postura_activa]
        else: # MODO_NORMAL o MODO_CONFIGURACION
            final_raw_angles, muneca = robot_logic.get_all_raw_angles(
                results_pose, results_hands, h, w, calibracion_completada,
                distancia_referencia, distancia_rotacion_referencia
            )
            gesture_buffer.append(final_raw_angles['pinza'])
            mano_estable = 1 if sum(gesture_buffer) >= config.GESTURE_CONFIRMATION_THRESHOLD else 0
            final_raw_angles['mano'] = mano_estable
            
            tiempo_restante_calibracion = 0
            if modo_actual == config.MODO_CONFIGURACION:
                if results_pose.pose_landmarks and results_hands.multi_hand_landmarks:
                    wrist_y = muneca[1]
                    zona_alto, start_y = h * 0.20, (h * 0.40)
                    if start_y <= wrist_y <= start_y + zona_alto:
                        if tiempo_inicio_calibracion == 0: tiempo_inicio_calibracion = time.time()
                        tiempo_transcurrido = time.time() - tiempo_inicio_calibracion
                        tiempo_restante_calibracion = max(0, config.CALIBRATION_TIME - tiempo_transcurrido)
                        if tiempo_transcurrido >= config.CALIBRATION_TIME:
                            hand_lm = results_hands.multi_hand_landmarks[0]
                            distancia_referencia = robot_logic._calcular_distancia_mano(hand_lm)
                            distancia_rotacion_referencia = robot_logic._calcular_distancia_mano(hand_lm, 5, 0)
                            calibracion_completada = True
                            print(f"Calibración completada. Distancia ref: {distancia_referencia:.4f}")
                            modo_actual = config.MODO_NORMAL
                    else:
                        tiempo_inicio_calibracion = 0
        
        mano_estable = final_raw_angles.get('mano', 0)
        angulos_finales = angle_processor.smooth_angles(final_raw_angles, test_servo_key=test_key)

        if arduino is not None and modo_actual != config.MODO_PAUSA:
            angulos_seguros = robot_logic.aplicar_limites_seguros(angulos_finales)
            datos = f"<{int(angulos_seguros['proximidad'])},{int(angulos_seguros['hombro'])},{int(angulos_seguros['codo'])},{int(angulos_seguros['pitch'])},{int(angulos_seguros['roll'])},{int(angulos_seguros['mano'])}>\n"
            print(f"Enviando a Arduino: {datos.strip()}")
            arduino.write(datos.encode('utf-8'))

        lienzo = np.zeros((h + 100, w + 450, 3), dtype=np.uint8)
        panel_sup = crear_panel_superior(w + 450, 100, logo_img)
        lienzo[0:100, 0:w+450] = panel_sup
        detector.draw_all_landmarks(frame, results_pose, results_hands)
        if modo_actual == config.MODO_CONFIGURACION:
            dibujar_zona_calibracion(frame, w, h)
        lienzo[100:100+h, 0:w] = frame
        panel_lat = crear_panel_lateral(450, h, angulos_finales, mano_estable, arduino is not None, modo_actual, servo_en_prueba, tiempo_restante_calibracion)
        lienzo[100:100+h, w:w+450] = panel_lat
        cv2.imshow("Control de Brazo Robotico", lienzo)
        
        key = cv2.waitKey(5) & 0xFF
        if key == 27: break

        if key == ord(' '):
            modo_actual, postura_activa, servo_en_prueba = config.MODO_CONFIGURACION, None, None
            tiempo_inicio_calibracion = 0; print("Modo cambiado a: CONFIGURACION")
        elif key == ord('n'):
            modo_actual, postura_activa, servo_en_prueba = config.MODO_NORMAL, None, None; print("Modo cambiado a: NORMAL")
        elif key == ord('a'):
            modo_actual, postura_activa, servo_en_prueba = config.MODO_POSTURA, 'saludo', None; print("Activando postura: 'saludo'")
        elif key == ord('p'):
            modo_actual = config.MODO_PAUSA; print("Modo cambiado a: PAUSA")
        elif ord('1') <= key <= ord('7'):
            servo_map = {'1':'proximidad','2':'hombro','3':'codo','4':'pitch','5':'roll','6':'mano', '7':'all'}
            servo_en_prueba = servo_map[chr(key)]
            modo_actual, postura_activa = config.MODO_PRUEBA, None; print(f"Modo cambiado a: PRUEBA - Servo: {servo_en_prueba}")

    cap.release()
    if arduino: arduino.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
