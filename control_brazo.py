# =================================================================
# PROYECTO: Control de Brazo Robótico con Visión (6 Ejes)
# VERSIÓN:  Final con Sensibilidad Aumentada
# =================================================================

import cv2  # Biblioteca para procesamiento de imágenes y visión por computadora
import mediapipe as mp  # Framework de detección de poses y manos
import numpy as np  # Biblioteca para cálculos numéricos y matrices
import time  # Para manejar tiempos y retardos
from collections import deque  # Estructura de datos tipo cola para buffers
import math  # Funciones matemáticas
import os  # Funciones del sistema operativo
import serial  # Comunicación serial con Arduino
import serial.tools.list_ports  # Herramientas para puertos seriales

# --- PARÁMETROS GLOBALES ---
# Estos valores definen el comportamiento del sistema
BASE_X_MIN_PORCENTAJE = 0.15  # Límite izquierdo para movimiento de base
BASE_X_MAX_PORCENTAJE = 0.85  # Límite derecho para movimiento de base
ROLL_INPUT_RANGE = 0.22  # Rango de entrada para la rotación de muñeca
ROLL_OUTPUT_MIN_ANGLE = 40  # Ángulo mínimo de salida para rotación
ROLL_OUTPUT_MAX_ANGLE = 130  # Ángulo máximo de salida para rotación
PITCH_INPUT_MIN_ANGLE = 150  # Ángulo mínimo para movimiento de pitch
PITCH_INPUT_MAX_ANGLE = 210  # Ángulo máximo para movimiento de pitch
GESTURE_BUFFER_SIZE = 10  # Tamaño del buffer para detección de gestos
GESTURE_CONFIRMATION_THRESHOLD = 7  # Umbral para confirmar gesto de pinza
SERIAL_PORT = 'COM3'  # Puerto serial para comunicación con Arduino
BAUD_RATE = 9600  # Velocidad de transmisión serial

# --- SISTEMA DE SEGURIDAD ESENCIAL ---
# Define los límites seguros para cada articulación del brazo robótico
ANGULOS_SEGUROS = {
    'base': (0, 180),      # Rango seguro para base (0-180°)
    'hombro': (30, 160),   # Rango seguro para hombro (30-160°)
    'codo': (20, 170),     # Rango seguro para codo (20-170°)
    'pitch': (0, 180),     # Rango seguro para movimiento vertical de muñeca
    'roll': (40, 130)      # Rango seguro para rotación de muñeca
}

# --- PALETA DE COLORES PARA ARTICULACIONES ---
# Asigna colores específicos a cada articulación para visualización
COLORES = {
    'base': (0, 0, 255),       # Rojo para base
    'hombro': (255, 255, 0),   # Turquesa para hombro
    'codo': (0, 165, 255),     # Naranja para codo
    'pitch': (0, 0, 255),      # Rojo para movimiento vertical de muñeca
    'roll': (150, 50, 200),    # Morado para rotación de muñeca
    'mano': (50, 50, 50)       # Gris para estado de la mano
}

# --- CLASE POSE DETECTOR ---
# Encapsula la funcionalidad de detección de poses y manos
class PoseDetector:
    def __init__(self):
        # Inicializa los módulos de dibujo y detección de MediaPipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        
        # Configura el detector de poses con niveles de confianza
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        
        # Configura el detector de manos (solo 1 mano)
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, 
            max_num_hands=1, 
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )

    # Procesa una imagen para detectar la pose corporal
    def find_pose(self, image): 
        return self.pose.process(image)
    
    # Procesa una imagen para detectar manos
    def find_hands(self, image): 
        return self.hands.process(image)
    
    # Dibuja todos los landmarks detectados en la imagen
    def draw_all_landmarks(self, image, pose_results, hand_results):
        # Si se detectó una pose, dibuja todos los landmarks y conexiones
        if pose_results.pose_landmarks: 
            self.mp_drawing.draw_landmarks(
                image, 
                pose_results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    color=(0, 0, 255), thickness=2, circle_radius=3
                )
            )
            
            # Obtiene dimensiones de la imagen
            h, w, _ = image.shape
            
            # Dibuja hombros en color turquesa
            for idx in [11, 12]:  # Índices de landmarks de hombros
                if idx < len(pose_results.pose_landmarks.landmark):
                    lm = pose_results.pose_landmarks.landmark[idx]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(image, (cx, cy), 8, (255, 255, 0), -1)
            
            # Dibuja codos en color naranja
            for idx in [13, 14]:  # Índices de landmarks de codos
                if idx < len(pose_results.pose_landmarks.landmark):
                    lm = pose_results.pose_landmarks.landmark[idx]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(image, (cx, cy), 8, (0, 165, 255), -1)
        
        # Si se detectaron manos, dibuja sus landmarks
        if hand_results.multi_hand_landmarks:
            for hand_lm in hand_results.multi_hand_landmarks: 
                self.mp_drawing.draw_landmarks(
                    image, 
                    hand_lm, 
                    self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                        color=(0, 0, 255), thickness=2, circle_radius=3
                    )
                )

# --- CÁLCULOS DE ÁNGULOS DEL BRAZO ---
# Calcula los ángulos de base, hombro y codo basados en los landmarks
def calcular_angulos_brazo(landmarks, h, w):
    # Obtiene coordenadas de hombro, codo y muñeca
    shoulder = [landmarks[12].x * w, landmarks[12].y * h]  # Landmark 12: hombro derecho
    elbow = [landmarks[14].x * w, landmarks[14].y * h]     # Landmark 14: codo derecho
    wrist = [landmarks[16].x * w, landmarks[16].y * h]     # Landmark 16: muñeca derecha

    # BASE: Calcula posición horizontal de la muñeca en el rango 0-180°
    base = np.interp(wrist[0], [w * BASE_X_MIN_PORCENTAJE, w * BASE_X_MAX_PORCENTAJE], [180, 0])

    # HOMBRO: Calcula el ángulo entre brazo y vertical
    vec_shoulder_elbow = [elbow[0] - shoulder[0], elbow[1] - shoulder[1]]
    eje_vertical = [0, -1]  # Vector vertical hacia arriba
    
    # Cálculo del producto punto para obtener el ángulo
    mag_vec = np.linalg.norm(vec_shoulder_elbow)
    dot = vec_shoulder_elbow[0] * eje_vertical[0] + vec_shoulder_elbow[1] * eje_vertical[1]

    if mag_vec > 0:
        cos_theta = dot / mag_vec
        cos_theta = max(min(cos_theta, 1), -1)  # Asegura valor válido [-1,1]
        ang_hombro = np.degrees(np.arccos(cos_theta))  # Convierte a grados
    else:
        ang_hombro = 90  # Valor por defecto

    # CODO: Calcula el ángulo entre antebrazo y brazo
    vec1 = [shoulder[0] - elbow[0], shoulder[1] - elbow[1]]  # Brazo
    vec2 = [wrist[0] - elbow[0], wrist[1] - elbow[1]]       # Antebrazo
    
    # Cálculo del ángulo entre los dos vectores
    dot_product = vec1[0]*vec2[0] + vec1[1]*vec2[1]
    mag1 = np.linalg.norm(vec1)
    mag2 = np.linalg.norm(vec2)

    if mag1 > 0 and mag2 > 0:
        cos_theta2 = dot_product / (mag1 * mag2)
        cos_theta2 = max(min(cos_theta2, 1), -1)
        ang_codo = np.degrees(np.arccos(cos_theta2))
    else:
        ang_codo = 90  # Valor por defecto

    return {'base': base, 'hombro': ang_hombro, 'codo': ang_codo}, elbow, wrist

# --- DETECCIÓN DE GESTOS DE MANO ---
# Detecta si la mano está haciendo el gesto de pinza
def detectar_pinza(hand_landmarks):
    # Puntos de referencia para dedos (punta y base)
    puntos = [
        (4, 2),  # Pulgar
        (8, 6),  # Índice
        (12, 10),  # Medio
        (16, 14),  # Anular
        (20, 18)   # Meñique
    ]
    
    # Cuenta cuántos dedos están flexionados
    flexionados = sum(
        1 for p1, p2 in puntos 
        if hand_landmarks.landmark[p1].y > hand_landmarks.landmark[p2].y
    )
    
    # Considera pinza cerrada si al menos 4 dedos están flexionados
    return 1 if flexionados >= 4 else 0

# Calcula los gestos de la mano (movimiento vertical y rotación)
def calcular_gestos_mano(hand_landmarks, codo, muneca):
    try:
        # PITCH: Movimiento vertical de muñeca
        # Calcula ángulo entre puntos de referencia de la mano
        ma_raw = math.degrees(
            math.atan2(hand_landmarks.landmark[5].y - muneca[1], 
                       hand_landmarks.landmark[5].x - muneca[0]) -
            math.atan2(codo[1] - muneca[1], codo[0] - muneca[0])
        )
        # Mapea a rango 0-180°
        ma = np.interp(abs(ma_raw), [PITCH_INPUT_MIN_ANGLE, PITCH_INPUT_MAX_ANGLE], [180, 0])
    except:
        ma = 90  # Valor por defecto en caso de error

    try:
        # ROLL: Rotación de muñeca
        # Calcula diferencia horizontal entre puntos de referencia
        p5_x = hand_landmarks.landmark[5].x  # Base del dedo índice
        p17_x = hand_landmarks.landmark[17].x  # Base del meñique
        
        # Mapea a rango 0-180° y luego a rango seguro
        mr_raw = np.interp(p5_x - p17_x, [-ROLL_INPUT_RANGE, ROLL_INPUT_RANGE], [180, 0])
        mr_limitado = np.interp(mr_raw, [0, 180], [ROLL_OUTPUT_MIN_ANGLE, ROLL_OUTPUT_MAX_ANGLE])
    except:
        mr_limitado = 90  # Valor por defecto

    try:
        # Detecta gesto de pinza
        p = detectar_pinza(hand_landmarks)
    except:
        p = 0  # Valor por defecto

    return {'pitch': ma, 'roll_raw': mr_limitado, 'pinza': p}

# --- APLICAR LÍMITES DE SEGURIDAD ---
# Restringe los ángulos a los rangos seguros definidos
def aplicar_limites_seguros(angulos):
    angulos_limitados = {}
    for eje, valor in angulos.items():
        if eje in ANGULOS_SEGUROS:
            min_val, max_val = ANGULOS_SEGUROS[eje]
            # Aplica límites: valor no menor que min_val, no mayor que max_val
            angulos_limitados[eje] = max(min_val, min(max_val, valor))
        else:
            angulos_limitados[eje] = valor
    return angulos_limitados

# --- PANEL SUPERIOR CON LOGO Y FONDO AZUL ---
# Crea el panel superior con logo y título
def crear_panel_superior(ancho_total, alto=80, logo_img=None):
    # Crea un panel con fondo azul (#00a1e0 en formato BGR)
    panel = np.zeros((alto, ancho_total, 3), dtype=np.uint8)
    panel[:] = (224, 161, 0)  # Color #00a1e0 en BGR (Blue=224, Green=161, Red=0)
    
    # Configura fuente para texto
    font = cv2.FONT_HERSHEY_SIMPLEX
    grosor = 1
    
    # Texto principal del panel
    texto = "Proyecto de brazo robotico"
    # Calcula tamaño del texto para centrarlo
    texto_size = cv2.getTextSize(texto, font, 1.1, grosor)[0]
    texto_x = (ancho_total - texto_size[0]) // 2
    # Dibuja texto centrado
    cv2.putText(panel, texto, (texto_x, 50), font, 1.1, (255, 255, 255), grosor)
    
    # Agrega logo si está disponible
    if logo_img is not None:
        try:
            # Calcula dimensiones manteniendo relación de aspecto
            logo_h, logo_w = alto - 20, int((alto - 20) * logo_img.shape[1] / logo_img.shape[0])
            logo_resized = cv2.resize(logo_img, (logo_w, logo_h))
            
            # Crea fondo blanco para el logo
            logo_bg = np.ones((logo_h, logo_w, 3), dtype=np.uint8) * 255
            logo_bg[0:logo_h, 0:logo_w] = logo_resized
            
            # Posiciona logo en la esquina izquierda
            panel[10:10+logo_h, 20:20+logo_w] = logo_bg
        except:
            pass  # Ignora errores en caso de problemas con el logo
    
    return panel

# --- PANEL LATERAL MODERNO ---
# Crea el panel lateral con información de articulaciones
def crear_panel_lateral(ancho, alto, angulos, mano_estable, conexion_serial):
    # Crea panel con fondo blanco
    panel = np.ones((alto, ancho, 3), dtype=np.uint8) * 255
    
    # Configura fuente para texto
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.8
    grosor = 1
    color_texto = (0, 0, 0)  # Texto negro
    
    # Estado de conexión serial
    estado_serial = "CONECTADO" if conexion_serial else "DESCONECTADO"
    color_serial = (0, 150, 0) if conexion_serial else (0, 0, 150)  # Verde o rojo
    # Muestra estado de conexión
    cv2.putText(panel, f"Serial: {estado_serial}", (20, 40), font, 0.7, color_serial, grosor)
    
    # Título de sección
    cv2.putText(panel, "ANGULOS DEL BRAZO", (20, 80), font, 0.9, (50, 50, 50), grosor)
    # Línea divisoria
    cv2.line(panel, (15, 90), (ancho - 15, 90), (200, 200, 200), 1)
    
    # Lista de articulaciones a mostrar
    articulaciones = [
        ('base', 'Base'),         # Articulación base
        ('hombro', 'Hombro'),     # Articulación de hombro
        ('codo', 'Codo'),         # Articulación de codo
        ('pitch', 'Muneca'),      # Movimiento vertical de muñeca
        ('roll', 'Rotacion'),     # Rotación de muñeca
        ('mano', 'Mano')          # Estado de la mano (abierta/cerrada)
    ]
    
    # Itera sobre cada articulación para mostrarla
    for i, (key, nombre) in enumerate(articulaciones):
        y_pos = 130 + i * 50  # Posición vertical para cada elemento
        
        # Punto de color representativo
        color = COLORES[key]  # Obtiene color de la paleta
        cv2.circle(panel, (30, y_pos), 8, color, -1)  # Dibuja punto
        
        # Obtiene valor a mostrar
        if key == 'mano':
            valor = "CERRADA" if mano_estable else "ABIERTA"  # Texto para mano
        else:
            valor = int(angulos[key])  # Valor numérico para otras articulaciones
        
        # Nombre de la articulación
        cv2.putText(panel, nombre, (50, y_pos + 5), font, font_scale, color_texto, grosor)
        # Valor de la articulación
        cv2.putText(panel, f"{valor}", (200, y_pos + 5), font, font_scale, color_texto, grosor)
    
    return panel

# --- FUNCIÓN PRINCIPAL ---
def main():
    # Carga el logo de la universidad
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

    # Inicializa conexión con Arduino
    arduino = None
    try:
        arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        time.sleep(1)  # Espera a que se establezca la conexión
        print(f"Serial conectado en {SERIAL_PORT}")
        conexion_serial = True
    except Exception as e:
        print(f"Error serial: {e}")
        arduino = None
        conexion_serial = False

    # Inicializa cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la camara")
        return

    # Crea detector de poses
    detector = PoseDetector()
    # Buffer para detección estable de gestos
    gesture_buffer = deque(maxlen=GESTURE_BUFFER_SIZE)
    panel_ancho = 350  # Ancho del panel lateral
    panel_alto_superior = 80  # Alto del panel superior

    # Bucle principal de procesamiento
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo capturar el frame")
            break
            
        frame = cv2.flip(frame, 1)  # Voltea horizontalmente para efecto espejo
        h, w, _ = frame.shape  # Obtiene dimensiones del frame
        
        # Crea lienzo principal (imagen completa)
        lienzo = np.zeros((h + panel_alto_superior, w + panel_ancho, 3), dtype=np.uint8)
        
        # Crea y añade panel superior
        panel_sup = crear_panel_superior(w + panel_ancho, panel_alto_superior, logo_img)
        lienzo[0:panel_alto_superior, 0:w+panel_ancho] = panel_sup
        
        # Convierte imagen a RGB para MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detecta pose corporal
        results_pose = detector.find_pose(image_rgb)
        # Detecta manos
        results_hands = detector.find_hands(image_rgb)

        # Valores por defecto para ángulos y puntos
        ang = {'base': 90, 'hombro': 90, 'codo': 90}
        gestos = {'pitch': 90, 'roll_raw': 90, 'pinza': 0}
        codo, muneca = [0, 0], [0, 0]

        # Si se detectó una pose, calcula ángulos
        if results_pose.pose_landmarks:
            ang, codo, muneca = calcular_angulos_brazo(results_pose.pose_landmarks.landmark, h, w)
            # Si se detectaron manos, calcula gestos
            if results_hands.multi_hand_landmarks:
                for hand in results_hands.multi_hand_landmarks:
                    try:
                        # Convierte coordenadas a relativas (0-1)
                        codo_rel = [codo[0] / w, codo[1] / h]
                        muneca_rel = [muneca[0] / w, muneca[1] / h]
                        gestos = calcular_gestos_mano(hand, codo_rel, muneca_rel)
                    except:
                        gestos = {'pitch': 90, 'roll_raw': 90, 'pinza': 0}

        # Actualiza buffer de gestos y determina estado estable de la mano
        gesture_buffer.append(gestos['pinza'])
        mano_estable = 1 if sum(gesture_buffer) >= GESTURE_CONFIRMATION_THRESHOLD else 0

        # Prepara diccionario con todos los ángulos
        angulos = {
            'base': ang['base'],
            'hombro': ang['hombro'],
            'codo': ang['codo'],
            'pitch': gestos['pitch'],
            'roll': gestos['roll_raw'],
            'mano': mano_estable
        }

        # Aplica límites de seguridad a los ángulos
        angulos_seguros = aplicar_limites_seguros(angulos)

        # Envía datos a Arduino si hay conexión
        if arduino is not None:
            try:
                # Formato: <base,hombro,codo,pitch,roll,mano>
                datos = f"<{int(angulos_seguros['base'])},{int(angulos_seguros['hombro'])},{int(angulos_seguros['codo'])},{int(angulos_seguros['pitch'])},{int(angulos_seguros['roll'])},{int(angulos_seguros['mano'])}>\n"
                arduino.write(datos.encode('utf-8'))
            except Exception as e:
                print(f"Error enviando datos: {e}")

        # Dibuja landmarks en el frame
        detector.draw_all_landmarks(frame, results_pose, results_hands)
        
        # Coloca el frame procesado en el lienzo principal
        lienzo[panel_alto_superior:panel_alto_superior+h, 0:w] = frame
        
        # Crea y añade panel lateral
        panel_lateral = crear_panel_lateral(panel_ancho, h, angulos, mano_estable, arduino is not None)
        lienzo[panel_alto_superior:panel_alto_superior+h, w:w+panel_ancho] = panel_lateral
        
        # Muestra la ventana con todo el contenido
        cv2.imshow("Control de Brazo Robotico", lienzo)
        
        # Termina el programa si se presiona ESC
        if cv2.waitKey(5) & 0xFF == 27: 
            break

    # Libera recursos al terminar
    cap.release()
    if arduino is not None:
        arduino.close()
    cv2.destroyAllWindows()

# Punto de entrada principal
if __name__ == "__main__":
    main()