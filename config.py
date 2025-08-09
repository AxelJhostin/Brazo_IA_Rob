# =================================================================
# MÓDULO: config.py
# DESCRIPCIÓN: Contiene todas las constantes y parámetros de
#               configuración para el proyecto del brazo robótico.
# VERSIÓN: 2.2 - Ajustada la sensibilidad del Pitch
# =================================================================

# --- CONFIGURACIÓN DE COMUNICACIÓN POR WI-FI (UDP) ---
ESP_IP = "192.168.100.155"  
ESP_PORT = 4210

# --- PARÁMETROS DE LA CÁMARA Y PROCESAMIENTO ---
CAMERA_INDEX = 0
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

# --- PARÁMETROS DE CONTROL Y GESTOS ---
GESTURE_BUFFER_SIZE = 10
GESTURE_CONFIRMATION_THRESHOLD = 7
CALIBRATION_TIME = 5

# --- PARÁMETROS DE LÓGICA DEL ROBOT (CÁLCULOS) ---
ANGLE_SMOOTHING_FACTOR = 0.1 
PROXIMITY_FILTER_FACTOR = 0.2 

ROLL_INPUT_RANGE = 0.22

# ¡CORRECCIÓN! Aumentamos el rango para un control más suave del pitch.
PITCH_INPUT_RANGE_MIN = -0.12 
PITCH_INPUT_RANGE_MAX = 0.12

ROLL_OUTPUT_MIN_ANGLE = 10
ROLL_OUTPUT_MAX_ANGLE = 180

DISTANCE_RANGE = 0.3

# --- PARÁMETROS DEL MODO DE PRUEBA ---
TEST_SWEEP_MIN = 10
TEST_SWEEP_MAX = 170
TEST_SWEEP_SPEED = 1.5
TEST_ALL_SERVOS_SPEED = 1.0

# --- LÍMITES DE SEGURIDAD DE LOS SERVOS (Grados) ---
# ¡CORRECCIÓN! Ajustamos el rango del pitch para un movimiento más estable.
ANGULOS_SEGUROS = {
    'proximidad': (0, 180), 'hombro': (30, 160), 'codo': (20, 170),
    'pitch': (30, 150), 'roll': (10, 180)
}

# --- POSTURAS PREDEFINIDAS ---
POSTURAS_PREDEFINIDAS = {
    'saludo': {
        'proximidad': 90, 'hombro': 120, 'codo': 90,
        'pitch': 90, 'roll': 90, 'mano': 0
    },
    'home': {
        'proximidad': 90, 'hombro': 90, 'codo': 90,
        'pitch': 90, 'roll': 90, 'mano': 0
    }
}

# --- PARÁMETROS DE LA INTERFAZ GRÁFICA ---
COLORES = {
    'proximidad': (0, 0, 255), 'hombro': (255, 255, 0), 'codo': (0, 165, 255),
    'pitch': (0, 0, 255), 'roll': (150, 50, 200), 'mano': (50, 50, 50)
}

# --- ESTADOS DEL PROGRAMA (Para claridad) ---
MODO_NORMAL = 0
MODO_CONFIGURACION = 1
MODO_POSTURA = 2
MODO_PRUEBA = 3
MODO_PAUSA = 4
MODO_GESTO_SI = 5
MODO_GESTO_NO = 6
MODO_MANUAL = 7
