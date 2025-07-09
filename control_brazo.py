# =================================================================
# PROYECTO: Control de Brazo Robótico con Visión (6 Ejes)
# VERSIÓN: Avanzada con Sistema de Proximidad Mejorado
# =================================================================

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import math
import os
import serial
import serial.tools.list_ports

# --- PARÁMETROS GLOBALES ---
BASE_X_MIN_PORCENTAJE = 0.15
BASE_X_MAX_PORCENTAJE = 0.85
ROLL_INPUT_RANGE = 0.22
ROLL_OUTPUT_MIN_ANGLE = 10  # Cambiado de 40 a 10 para mejor rango
ROLL_OUTPUT_MAX_ANGLE = 180  # Cambiado de 130 a 180 para mejor rango
PITCH_INPUT_MIN_ANGLE = 150
PITCH_INPUT_MAX_ANGLE = 210
GESTURE_BUFFER_SIZE = 10
GESTURE_CONFIRMATION_THRESHOLD = 7
SERIAL_PORT = 'COM3'
BAUD_RATE = 9600
CALIBRATION_TIME = 5  # Tiempo para calibración en segundos
DISTANCE_THRESHOLD = 0.1  # Umbral del 10% para cambios significativos
DISTANCE_RANGE = 0.3  # Rango de distancias para mapeo a 0-180 grados
PROXIMITY_FILTER_FACTOR = 0.2  # Factor de filtrado para valores de proximidad

# --- SISTEMA DE SEGURIDAD ---
ANGULOS_SEGUROS = {
    'hombro': (30, 160),
    'codo': (20, 170),
    'pitch': (0, 180),
    'roll': (10, 180),  # Ajustado al nuevo rango
    'proximidad': (0, 180)  # Nuevo rango para el valor de proximidad
}

# --- PALETA DE COLORES ---
COLORES = {
    'proximidad': (0, 0, 255),  # Rojo para proximidad
    'hombro': (255, 255, 0),    # Turquesa
    'codo': (0, 165, 255),      # Naranja
    'pitch': (0, 0, 255),       # Rojo
    'roll': (150, 50, 200),     # Morado
    'mano': (50, 50, 50)        # Gris
}

# --- ESTADOS DEL PROGRAMA ---
MODO_NORMAL = 0
MODO_CONFIGURACION = 1
modo_actual = MODO_NORMAL
calibracion_completada = False
distancia_referencia = 0
distancia_rotacion_referencia = 0  # Nueva referencia para rotación
tiempo_inicio_calibracion = 0
distancia_actual = ""
valor_proximidad_filtrado = 90  # Valor filtrado para suavizar movimientos

# --- CLASE POSE DETECTOR CON VISUALIZACIÓN MEJORADA ---
class PoseDetector:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        
        # Configuración de alta calidad para MediaPipe
        self.pose = self.mp_pose.Pose(
            model_complexity=2,  # Modelo complejo para mejor precisión
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
        # Dibujar landmarks de pose con conexiones visibles
        if pose_results.pose_landmarks: 
            # Grosor aumentado para mejor visibilidad
            self.mp_drawing.draw_landmarks(
                image, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    color=(0, 0, 255), thickness=3, circle_radius=5
                ),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    color=(180, 180, 180), thickness=3  # Grosor aumentado
                )
            )
            
            h, w, _ = image.shape
            
            # Hombros en turquesa (más grandes)
            for idx in [11, 12]:
                if idx < len(pose_results.pose_landmarks.landmark):
                    lm = pose_results.pose_landmarks.landmark[idx]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(image, (cx, cy), 12, (255, 255, 0), -1)
            
            # Codos en naranja (más grandes)
            for idx in [13, 14]:
                if idx < len(pose_results.pose_landmarks.landmark):
                    lm = pose_results.pose_landmarks.landmark[idx]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(image, (cx, cy), 12, (0, 165, 255), -1)
        
        # Dibujar landmarks de mano con conexiones visibles
        if hand_results.multi_hand_landmarks:
            for hand_lm in hand_results.multi_hand_landmarks: 
                self.mp_drawing.draw_landmarks(
                    image, hand_lm, self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                        color=(0, 0, 255), thickness=3, circle_radius=5
                    ),
                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                        color=(180, 180, 180), thickness=2
                    )
                )

# --- CÁLCULOS DE ÁNGULOS ---
def calcular_angulos_brazo(landmarks, h, w):
    shoulder = [landmarks[12].x * w, landmarks[12].y * h]
    elbow = [landmarks[14].x * w, landmarks[14].y * h]
    wrist = [landmarks[16].x * w, landmarks[16].y * h]

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

    return {'hombro': ang_hombro, 'codo': ang_codo}, elbow, wrist

# --- DETECCIÓN DE GESTOS Y DISTANCIA ---
def detectar_pinza(hand_landmarks):
    puntos = [
        (4, 2), (8, 6), (12, 10), (16, 14), (20, 18)
    ]
    flexionados = sum(
        1 for p1, p2 in puntos 
        if hand_landmarks.landmark[p1].y > hand_landmarks.landmark[p2].y
    )
    return 1 if flexionados >= 4 else 0

def calcular_distancia_mano(hand_landmarks, punto1=5, punto2=17):
    """Calcula la distancia entre dos landmarks de la mano"""
    # Puntos para cálculo de distancia (por defecto 5 y 17)
    punto_a = hand_landmarks.landmark[punto1]
    punto_b = hand_landmarks.landmark[punto2]
    
    # Distancia euclidiana entre puntos
    distancia = math.sqrt(
        (punto_a.x - punto_b.x)**2 + 
        (punto_a.y - punto_b.y)**2
    )
    return distancia

def calcular_gestos_mano(hand_landmarks, codo, muneca, distancia_rotacion_ref):
    try:
        # Cálculo del ángulo de pitch (muñeca)
        ma_raw = math.degrees(math.atan2(hand_landmarks.landmark[5].y - muneca[1], 
                                         hand_landmarks.landmark[5].x - muneca[0]) -
                          math.atan2(codo[1] - muneca[1], codo[0] - muneca[0]))
        ma = np.interp(abs(ma_raw), [PITCH_INPUT_MIN_ANGLE, PITCH_INPUT_MAX_ANGLE], [180, 0])
    except:
        ma = 90

    try:
        # Cálculo del ángulo de roll (rotación) mejorado
        # Primero calculamos la distancia entre punto 5 y 0 (referencia de rotación)
        distancia_actual_rotacion = calcular_distancia_mano(hand_landmarks, 5, 0)
        
        # Si tenemos una referencia de rotación, usamos esa para calcular el roll
        if distancia_rotacion_ref > 0:
            # Calculamos la variación respecto a la referencia
            variacion_rotacion = distancia_actual_rotacion - distancia_rotacion_ref
            # Mapeamos la variación al rango de roll (10-180)
            mr_limitado = np.interp(variacion_rotacion, 
                                  [-ROLL_INPUT_RANGE, ROLL_INPUT_RANGE], 
                                  [10, 180])
        else:
            # Si no hay referencia, usamos el método anterior
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

# --- APLICAR LÍMITES DE SEGURIDAD ---
def aplicar_limites_seguros(angulos):
    angulos_limitados = {}
    for eje, valor in angulos.items():
        if eje in ANGULOS_SEGUROS:
            min_val, max_val = ANGULOS_SEGUROS[eje]
            angulos_limitados[eje] = max(min_val, min(max_val, valor))
        else:
            angulos_limitados[eje] = valor
    return angulos_limitados

# --- CALCULAR VALOR DE PROXIMIDAD CON FILTRADO ---
def calcular_proximidad(distancia_actual, distancia_referencia):
    """Calcula el valor de proximidad (0-180) basado en la distancia con filtrado"""
    global valor_proximidad_filtrado
    
    if not calibracion_completada or distancia_referencia == 0:
        return 90
    
    # Calcular umbral absoluto (10% de la distancia de referencia)
    umbral_abs = distancia_referencia * DISTANCE_THRESHOLD
    
    # Determinar rango de distancias
    dist_min = distancia_referencia - (distancia_referencia * DISTANCE_RANGE)
    dist_max = distancia_referencia + (distancia_referencia * DISTANCE_RANGE)
    
    # Mapear a rango 0-180
    if distancia_actual <= dist_min:
        valor_raw = 0
    elif distancia_actual >= dist_max:
        valor_raw = 180
    else:
        valor_raw = np.interp(distancia_actual, [dist_min, dist_max], [0, 180])
    
    # Aplicar filtro de suavizado
    valor_proximidad_filtrado = (PROXIMITY_FILTER_FACTOR * valor_raw) + ((1 - PROXIMITY_FILTER_FACTOR) * valor_proximidad_filtrado)
    
    return int(valor_proximidad_filtrado)

# --- PANEL SUPERIOR CON LOGO ---
def crear_panel_superior(ancho_total, alto=90, logo_img=None):
    panel = np.zeros((alto, ancho_total, 3), dtype=np.uint8)
    panel[:] = (224, 161, 0)  # Azul #00a1e0
    
    # Fuente mejorada
    font = cv2.FONT_HERSHEY_DUPLEX
    grosor = 2  # Aumentado para mejor visibilidad
    escala = 1.2
    
    texto = "PROYECTO BRAZO ROBOTICO"  # Todo en mayúsculas
    texto_size = cv2.getTextSize(texto, font, escala, grosor)[0]
    texto_x = (ancho_total - texto_size[0]) // 2
    texto_y = 60
    
    # Texto con sombra para mejor contraste
    cv2.putText(panel, texto, (texto_x+2, texto_y+2), font, escala, (0, 0, 0), grosor+1)
    cv2.putText(panel, texto, (texto_x, texto_y), font, escala, (255, 255, 255), grosor)
    
    # Logo con fondo blanco en lado izquierdo
    if logo_img is not None:
        try:
            # Aumentar tamaño del logo
            logo_h = alto - 30
            logo_w = int(logo_h * logo_img.shape[1] / logo_img.shape[0])
            
            logo_resized = cv2.resize(logo_img, (logo_w, logo_h))
            
            # Crear fondo blanco para el logo
            logo_bg = np.ones((logo_h, logo_w, 3), dtype=np.uint8) * 255
            logo_bg[0:logo_h, 0:logo_w] = logo_resized
            
            # Posicionar en esquina izquierda con margen
            panel[15:15+logo_h, 30:30+logo_w] = logo_bg
        except Exception as e:
            print(f"Error dibujando logo: {str(e)}")
    
    return panel

# --- PANEL LATERAL CON SISTEMA DE PROXIMIDAD ---
def crear_panel_lateral(ancho, alto, angulos, mano_estable, conexion_serial, modo_configuracion=False, tiempo_restante=0):
    panel = np.ones((alto, ancho, 3), dtype=np.uint8) * 255
    
    # Fuentes mejoradas
    font_titulo = cv2.FONT_HERSHEY_DUPLEX
    font_modo = cv2.FONT_HERSHEY_DUPLEX
    font_texto = cv2.FONT_HERSHEY_SIMPLEX  # Fuente más legible
    font_valores = cv2.FONT_HERSHEY_SIMPLEX  # Fuente para valores
    
    grosor_normal = 2  # Bold para texto
    grosor_modo = 2    # Bold para modo
    grosor_valores = 2  # Ahora también en bold
    
    # Estado de conexión
    estado_serial = "CONECTADO" if conexion_serial else "DESCONECTADO"
    color_serial = (0, 180, 0) if conexion_serial else (0, 0, 180)
    cv2.putText(panel, f"SERIAL: {estado_serial}", (30, 50), font_texto, 0.9, color_serial, grosor_normal)
    
    # Indicador de modo
    if modo_configuracion:
        # Fondo oscuro para modo
        cv2.rectangle(panel, (20, 70), (ancho-20, 130), (40, 40, 40), -1)
        
        # Texto principal de modo
        cv2.putText(panel, "MODO CONFIGURACION", 
                   (ancho//2 - 220, 110), font_modo, 1.1, (0, 255, 255), grosor_modo)
        
        if tiempo_restante > 0:
            # Texto de calibración con fondo claro
            cv2.rectangle(panel, (20, 140), (ancho-20, 180), (240, 240, 200), -1)
            cv2.putText(panel, f"CALIBRANDO: {tiempo_restante:.1f}s", 
                       (ancho//2 - 150, 170), font_texto, 0.9, (0, 0, 0), grosor_normal)
        else:
            # Instrucciones con fondo claro
            cv2.rectangle(panel, (20, 140), (ancho-20, 180), (240, 240, 200), -1)
            cv2.putText(panel, "COLOCAR BRAZO EN ZONA AMARILLA", 
                       (ancho//2 - 250, 170), font_texto, 0.8, (0, 0, 0), grosor_normal)
        
        cv2.line(panel, (25, 190), (ancho - 25, 190), (200, 200, 200), 2)
        y_start = 220
    else:
        # Diseño para modo normal
        cv2.rectangle(panel, (20, 70), (ancho-20, 130), (40, 40, 40), -1)
        cv2.putText(panel, "MODO OPERACION", 
                   (ancho//2 - 150, 110), font_modo, 1.1, (0, 255, 0), grosor_modo)
        cv2.line(panel, (25, 140), (ancho - 25, 140), (200, 200, 200), 2)
        y_start = 160
    
    # Título de sección para articulaciones con más espacio
    cv2.putText(panel, "ESTADO DE ARTICULACIONES", 
               (ancho//2 - 200, y_start+40), font_texto, 0.9, (60, 60, 60), grosor_normal)
    y_start += 80  # Aumentado el espacio a 80 píxeles (antes era 60)
    
    # Articulaciones con diseño mejorado
    articulaciones = [
        ('proximidad', 'PROXIMIDAD'),
        ('hombro', 'HOMBRO'),
        ('codo', 'CODO'),
        ('pitch', 'MUÑECA'),
        ('roll', 'ROTACION'),
        ('mano', 'MANO')
    ]
    
    for i, (key, nombre) in enumerate(articulaciones):
        y_pos = y_start + i * 60
        
        color = COLORES[key]
        cv2.circle(panel, (40, y_pos), 12, color, -1)
        
        if modo_configuracion:
            valor = "-"  # No mostrar valores durante calibración
        else:
            valor = int(angulos[key]) if key != 'mano' else ("CERRADA" if mano_estable else "ABIERTA")
        
        # Texto en bold
        cv2.putText(panel, nombre, (80, y_pos + 10), font_texto, 0.9, (0, 0, 0), grosor_normal)
        # Valores también en bold
        cv2.putText(panel, f"{valor}", (280, y_pos + 10), font_valores, 0.9, (0, 0, 0), grosor_valores)
    
    return panel

# --- DIBUJAR ZONA DE CALIBRACIÓN HORIZONTAL ---
def dibujar_zona_calibracion(frame, ancho, alto):
    # Franja horizontal amarilla en el centro (20% del alto)
    zona_alto = int(alto * 0.20)
    start_y = (alto - zona_alto) // 2
    end_y = start_y + zona_alto
    
    # Dibujar rectángulo semitransparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, start_y), (ancho, end_y), (0, 255, 255), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # Texto indicativo mejorado
    cv2.putText(frame, "ZONA DE CALIBRACION", (ancho//2 - 220, start_y - 30), 
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 0, 0), 3)
    cv2.putText(frame, "Mantenga el brazo recto aqui", (ancho//2 - 220, start_y + 40), 
                cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 0, 0), 2)

# --- VERIFICAR SI LA MUÑECA ESTÁ EN ZONA DE CALIBRACIÓN ---
def en_zona_calibracion(wrist_y, alto):
    zona_alto = alto * 0.20
    start_y = (alto - zona_alto) // 2
    end_y = start_y + zona_alto
    return start_y <= wrist_y <= end_y

# --- MAIN CON SISTEMA DE PROXIMIDAD MEJORADO ---
def main():
    global modo_actual, calibracion_completada, distancia_referencia, tiempo_inicio_calibracion, distancia_actual, valor_proximidad_filtrado, distancia_rotacion_referencia
    
    # Configuración de ventana redimensionable
    cv2.namedWindow("Control de Brazo Robotico", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Control de Brazo Robotico", 1600, 900)
    
    # Cargar logo
    logo_img = None
    try:
        logo_path = "logo_puce.png"
        if os.path.exists(logo_path):
            logo_img = cv2.imread(logo_path)
            if logo_img is not None:
                print("Logo PUCE cargado")
            else:
                print("Error: no se pudo cargar el logo")
        else:
            print("Archivo de logo no encontrado")
    except Exception as e:
        print(f"Error cargando logo: {str(e)}")
        logo_img = None

    # Inicializar conexión serial
    arduino = None
    try:
        arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        time.sleep(1)
        print(f"Serial conectado en {SERIAL_PORT}")
        conexion_serial = True
    except Exception as e:
        print(f"Error serial: {e}")
        arduino = None
        conexion_serial = False

    # Configurar cámara en alta resolución
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la camara")
        return
    
    # Configurar máxima resolución disponible
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # Verificar resolución real obtenida
    ancho_cam = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto_cam = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolucion de camara: {ancho_cam}x{alto_cam}")

    detector = PoseDetector()
    gesture_buffer = deque(maxlen=GESTURE_BUFFER_SIZE)
    panel_ancho = 450
    panel_alto_superior = 100
    tiempo_restante_calibracion = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo capturar el frame")
            break
            
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Crear lienzo principal con espacio para panel superior
        lienzo = np.zeros((h + panel_alto_superior, w + panel_ancho, 3), dtype=np.uint8)
        
        # Crear panel superior
        panel_sup = crear_panel_superior(w + panel_ancho, panel_alto_superior, logo_img)
        lienzo[0:panel_alto_superior, 0:w+panel_ancho] = panel_sup
        
        # Procesamiento de imagen
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_pose = detector.find_pose(image_rgb)
        results_hands = detector.find_hands(image_rgb)

        ang = {'hombro': 90, 'codo': 90}
        gestos = {'pitch': 90, 'roll_raw': 90, 'pinza': 0}
        codo, muneca = [0, 0], [0, 0]
        wrist_y = 0
        distancia_mano_actual = 0
        distancia_rotacion_actual = 0

        # Variables para detección de mano
        mano_detectada = False
        valor_proximidad = 90  # Valor por defecto
        
        if results_pose.pose_landmarks:
            ang, codo, muneca = calcular_angulos_brazo(results_pose.pose_landmarks.landmark, h, w)
            wrist_y = muneca[1]  # Posición Y de la muñeca para calibración
            
            if results_hands.multi_hand_landmarks:
                for hand in results_hands.multi_hand_landmarks:
                    try:
                        codo_rel = [codo[0] / w, codo[1] / h]
                        muneca_rel = [muneca[0] / w, muneca[1] / h]
                        
                        # Calcular distancia de la mano (entre landmarks 5 y 17)
                        distancia_mano_actual = calcular_distancia_mano(hand)
                        # Calcular distancia para rotación (entre 5 y 0)
                        distancia_rotacion_actual = calcular_distancia_mano(hand, 5, 0)
                        
                        # Calcular gestos con la referencia de rotación
                        gestos = calcular_gestos_mano(hand, codo_rel, muneca_rel, distancia_rotacion_referencia)
                        mano_detectada = True
                    except:
                        gestos = {'pitch': 90, 'roll_raw': 90, 'pinza': 0}

        gesture_buffer.append(gestos['pinza'])
        mano_estable = 1 if sum(gesture_buffer) >= GESTURE_CONFIRMATION_THRESHOLD else 0

        # Manejo del modo configuración
        if modo_actual == MODO_CONFIGURACION:
            # Dibujar zona de calibración HORIZONTAL
            dibujar_zona_calibracion(frame, w, h)
            
            # Verificar si la muñeca está en la zona de calibración
            if en_zona_calibracion(wrist_y, h):
                if tiempo_inicio_calibracion == 0:
                    tiempo_inicio_calibracion = time.time()
                
                tiempo_transcurrido = time.time() - tiempo_inicio_calibracion
                tiempo_restante_calibracion = max(0, CALIBRATION_TIME - tiempo_transcurrido)
                
                # Completar calibración si ha pasado el tiempo suficiente
                if tiempo_transcurrido >= CALIBRATION_TIME:
                    if mano_detectada:
                        distancia_referencia = distancia_mano_actual
                        distancia_rotacion_referencia = distancia_rotacion_actual  # Guardar referencia de rotación
                        calibracion_completada = True
                        print(f"Calibracion completada. Distancia referencia: {distancia_referencia:.4f}")
                        print(f"Referencia de rotacion: {distancia_rotacion_referencia:.4f}")
                    modo_actual = MODO_NORMAL
                    tiempo_inicio_calibracion = 0
            else:
                # Reiniciar temporizador si sale de la zona
                tiempo_inicio_calibracion = 0
                tiempo_restante_calibracion = 0
        else:
            # Calcular valor de proximidad en modo normal
            if calibracion_completada and mano_detectada:
                valor_proximidad = calcular_proximidad(distancia_mano_actual, distancia_referencia)

        # Obtener ángulos (con valor de proximidad reemplazando base)
        angulos = {
            'proximidad': valor_proximidad,
            'hombro': ang['hombro'],
            'codo': ang['codo'],
            'pitch': gestos['pitch'],
            'roll': gestos['roll_raw'],
            'mano': mano_estable
        }

        # Enviar datos a Arduino solo en modo normal
        if modo_actual == MODO_NORMAL and arduino is not None:
            try:
                angulos_seguros = aplicar_limites_seguros(angulos)
                datos = f"<{int(angulos_seguros['proximidad'])},{int(angulos_seguros['hombro'])},{int(angulos_seguros['codo'])},{int(angulos_seguros['pitch'])},{int(angulos_seguros['roll'])},{int(angulos_seguros['mano'])}>\n"
                arduino.write(datos.encode('utf-8'))
            except Exception as e:
                print(f"Error enviando datos: {e}")

        # Dibujar landmarks (con líneas visibles)
        detector.draw_all_landmarks(frame, results_pose, results_hands)
        
        # Posicionar frame en el lienzo
        lienzo[panel_alto_superior:panel_alto_superior+h, 0:w] = frame
        
        # Crear y posicionar panel lateral
        panel_lateral = crear_panel_lateral(
            panel_ancho, h, 
            angulos, mano_estable, 
            arduino is not None,
            modo_actual == MODO_CONFIGURACION,
            tiempo_restante_calibracion
        )
        lienzo[panel_alto_superior:panel_alto_superior+h, w:w+panel_ancho] = panel_lateral
        
        # Mostrar ventana
        cv2.imshow("Control de Brazo Robotico", lienzo)
        
        # Manejo de teclas
        key = cv2.waitKey(5) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # Barra espaciadora
            if modo_actual == MODO_NORMAL:
                modo_actual = MODO_CONFIGURACION
                tiempo_inicio_calibracion = 0
                print("Modo configuracion activado")
            else:
                modo_actual = MODO_NORMAL
                print("Modo configuracion cancelado")
        
        # Ajustar ventana si es necesario (responsive)
        win_h, win_w = lienzo.shape[:2]
        cv2.resizeWindow("Control de Brazo Robotico", win_w, win_h)

    # Liberar recursos
    cap.release()
    if arduino is not None:
        arduino.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()