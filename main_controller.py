# =================================================================
# PROYECTO: Control de Brazo Robótico con Visión (10 Ejes)
# VERSIÓN: 3.0 - Final para mano de 5 dedos
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

def encontrar_puerto_arduino():
    """Busca y devuelve el puerto serial donde está conectado el Arduino."""
    print("Buscando Arduino...")
    puertos = serial.tools.list_ports.comports()
    for puerto in puertos:
        # Puedes ajustar esto si tu placa se identifica diferente
        if "Arduino" in puerto.description or "CH340" in puerto.description:
            print(f"Arduino encontrado en: {puerto.device}")
            return puerto.device
    
    print("ADVERTENCIA: No se encontró un Arduino automáticamente.")
    if puertos:
        print(f"Usando el primer puerto disponible: {puertos[0].device}")
        return puertos[0].device
    return None

def main():
    # --- INICIALIZACIÓN ---
    modo_actual = config.MODO_NORMAL
    postura_activa, servo_en_prueba = None, None
    calibracion_completada = False
    distancia_referencia, distancia_rotacion_referencia = 0, 0
    
    # --- Variables para el Modo Manual ---
    articulacion_seleccionada_idx = 0
    manual_angles = {}
    articulaciones_keys = ['proximidad', 'hombro', 'codo', 'pitch', 'roll', 'pulgar', 'indice', 'medio', 'anular', 'menique']
    
    angle_processor = robot_logic.AngleProcessor()

    cv2.namedWindow("Control de Brazo Robotico", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Control de Brazo Robotico", 1600, 900)
    logo_img = cv2.imread("logo_puce.png") if os.path.exists("logo_puce.png") else None
    
    # --- Conexión Serial ---
    puerto_arduino = encontrar_puerto_arduino()
    ser = None
    conexion_brazo = False
    
    if puerto_arduino:
        try:
            # Conecta con el Arduino
            ser = serial.Serial(puerto_arduino, config.SERIAL_BAUDRATE, timeout=1)
            # Espera 2 segundos a que el Arduino se reinicie (importante)
            time.sleep(2) 
            conexion_brazo = True
            print(f"Conectado al Arduino en {puerto_arduino} a {config.SERIAL_BAUDRATE} baudios.")
        except serial.SerialException as e:
            print(f"Error al conectar por serial: {e}")
    else:
        print("Error: No se encontró ningún puerto serial disponible.")

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened(): print("Error: No se pudo abrir la cámara"); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    
    detector = robot_logic.PoseDetector()

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
        
        # Lógica de cálculo de ángulos por modo
        if modo_actual == config.MODO_MANUAL:
            final_raw_angles = manual_angles
        elif modo_actual == config.MODO_PRUEBA:
            test_key = servo_en_prueba
        elif modo_actual == config.MODO_POSTURA and postura_activa:
            final_raw_angles = config.POSTURAS_PREDEFINIDAS[postura_activa].copy()
        else: # MODO_NORMAL, MODO_GESTO, etc., todos requieren la detección
            final_raw_angles, muneca = robot_logic.get_all_raw_angles(
                results_pose, results_hands, h, w, calibracion_completada,
                distancia_referencia, distancia_rotacion_referencia
            )
        
        # Procesado y suavizado de ángulos
        angulos_finales = angle_processor.smooth_angles(final_raw_angles, hand_results=results_hands, test_servo_key=test_key, modo_actual=modo_actual)
        
        if modo_actual == config.MODO_MANUAL:
            manual_angles = angulos_finales.copy()

        # Envío de datos al robot
        if conexion_brazo and modo_actual != config.MODO_PAUSA:
            angulos_seguros = robot_logic.aplicar_limites_seguros(angulos_finales)
            
            # Aseguramos que todos los valores existan antes de enviarlos
            datos = (f"<{int(angulos_seguros.get('proximidad', 90))},"
                     f"{int(angulos_seguros.get('hombro', 90))},"
                     f"{int(angulos_seguros.get('codo', 90))},"
                     f"{int(angulos_seguros.get('pitch', 90))},"
                     f"{int(angulos_seguros.get('roll', 90))},"
                     f"{int(angulos_seguros.get('pulgar', 90))},"
                     f"{int(angulos_seguros.get('indice', 90))},"
                     f"{int(angulos_seguros.get('medio', 90))},"
                     f"{int(angulos_seguros.get('anular', 90))},"
                     f"{int(angulos_seguros.get('menique', 90))}>")
            
            # ¡CAMBIO CLAVE! Enviar por serial en lugar de socket
            ser.write(datos.encode('utf-8'))

        # Creación de la interfaz gráfica
        lienzo = np.zeros((h + 100, w + 450, 3), dtype=np.uint8)
        panel_sup = crear_panel_superior(w + 450, 100, logo_img)
        lienzo[0:100, 0:w+450] = panel_sup
        detector.draw_all_landmarks(frame, results_pose, results_hands)
        lienzo[100:100+h, 0:w] = frame
        
        # CORRECCIÓN: Se elimina 'mano_estable' de la llamada
        panel_lat = crear_panel_lateral(450, h, angulos_finales, conexion_brazo, modo_actual, servo_en_prueba, 0, postura_activa, articulacion_seleccionada_idx)
        lienzo[100:100+h, w:w+450] = panel_lat
        cv2.imshow("Control de Brazo Robotico", lienzo)
        
        # Captura de teclado
        key = cv2.waitKeyEx(5)

        if key != -1:
            if key == 27: break

            # Lógica del Modo Manual (simplificada)
            if modo_actual == config.MODO_MANUAL:
                key_str = articulaciones_keys[articulacion_seleccionada_idx]
                if key == 2490368: # Flecha Arriba
                    articulacion_seleccionada_idx = (articulacion_seleccionada_idx - 1) % len(articulaciones_keys)
                elif key == 2621440: # Flecha Abajo
                    articulacion_seleccionada_idx = (articulacion_seleccionada_idx + 1) % len(articulaciones_keys)
                elif key == 2424832: # Flecha Izquierda
                    manual_angles[key_str] = manual_angles.get(key_str, 90) - 2
                elif key == 2555904: # Flecha Derecha
                    manual_angles[key_str] = manual_angles.get(key_str, 90) + 2
            
            # Lógica de cambio de modo
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
                # CORRECCIÓN: Se elimina la 'mano' y se reduce el rango de teclas
                servo_map = {'1':'proximidad','2':'hombro','3':'codo','4':'pitch','5':'roll', '7':'all'}
                servo_en_prueba = servo_map.get(chr(key))
                if servo_en_prueba:
                    modo_actual, postura_activa = config.MODO_PRUEBA, None
                    print(f"Modo cambiado a: PRUEBA - Servo: {servo_en_prueba}")

    cap.release()
    if ser: ser.close() 
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()