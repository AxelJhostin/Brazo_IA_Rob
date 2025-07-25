# =================================================================
# MÓDULO: config.py
# DESCRIPCIÓN: Contiene todas las constantes y parámetros de
# configuración para el proyecto del brazo robótico.
# =================================================================

# --- CONFIGURACIÓN DE COMUNICACIÓN SERIAL ---
SERIAL_PORT = 'COM4'
BAUD_RATE = 9600

# --- PARÁMETROS DE LA CÁMARA Y PROCESAMIENTO ---
CAMERA_INDEX = 0
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

# --- PARÁMETROS DE CONTROL Y GESTOS ---
GESTURE_BUFFER_SIZE = 10
GESTURE_CONFIRMATION_THRESHOLD = 7
CALIBRATION_TIME = 5

# --- PARÁMETROS DE LÓGICA DEL ROBOT (CÁLCULOS) ---
# NUEVO: Factor de suavizado para los ángulos principales. Más bajo = más suave.
ANGLE_SMOOTHING_FACTOR = 0.3 
PROXIMITY_FILTER_FACTOR = 0.2 # Mantenemos uno separado para la base por si se quiere diferente

ROLL_INPUT_RANGE = 0.22
ROLL_OUTPUT_MIN_ANGLE = 10
ROLL_OUTPUT_MAX_ANGLE = 180
PITCH_INPUT_MIN_ANGLE = 150
PITCH_INPUT_MAX_ANGLE = 210
DISTANCE_RANGE = 0.3

# --- LÍMITES DE SEGURIDAD DE LOS SERVOS (Grados) ---
ANGULOS_SEGUROS = {
    'hombro': (30, 160),
    'codo': (20, 170),
    'pitch': (0, 180),
    'roll': (10, 180),
    'proximidad': (0, 180)
}

# --- PARÁMETROS DE LA INTERFAZ GRÁFICA ---
COLORES = {
    'proximidad': (0, 0, 255),
    'hombro': (255, 255, 0),
    'codo': (0, 165, 255),
    'pitch': (0, 0, 255),
    'roll': (150, 50, 200),
    'mano': (50, 50, 50)
}

# --- ESTADOS DEL PROGRAMA (Para claridad) ---
MODO_NORMAL = 0
MODO_CONFIGURACION = 1
