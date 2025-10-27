# =================================================================
# MÓDULO: ui_components.py (versión mejorada)
# DESCRIPCIÓN: Interfaz gráfica del controlador del brazo robótico
# RESPETA colores institucionales ya existentes
# =================================================================

import numpy as np
import cv2
import config

# -----------------------------
# Estilo general
# -----------------------------
COLOR_FONDO_PANEL = (250, 250, 250)
COLOR_TEXTO_OSCURO = (40, 40, 40)
COLOR_TEXTO_CLARO = (255, 255, 255)
COLOR_RESALTADO = (0, 255, 255)
COLOR_BARRA_DEFAULT = (255, 191, 0)  # dorado suave (ya usado)

# -----------------------------
# Funciones de dibujo
# -----------------------------

def _dibujar_barra_angulo_reducida(panel_fila, y, nombre, valor, color, seleccionado=False):
    """Dibuja una etiqueta, barra y valor en un espacio vertical más limpio y moderno."""
    font_texto = cv2.FONT_HERSHEY_SIMPLEX
    font_valores = cv2.FONT_HERSHEY_DUPLEX
    ancho_panel = panel_fila.shape[1]

    # Fondo de selección suave
    if seleccionado:
        overlay = panel_fila.copy()
        cv2.rectangle(overlay, (5, y + 2), (ancho_panel - 5, y + 40), (0, 255, 255), -1)
        cv2.addWeighted(overlay, 0.25, panel_fila, 0.75, 0, panel_fila)
        cv2.rectangle(panel_fila, (5, y + 2), (ancho_panel - 5, y + 40), (0, 200, 200), 2)

    # Nombre
    cv2.putText(panel_fila, nombre, (20, y + 28), font_texto, 0.6, COLOR_TEXTO_OSCURO, 2)

    # Barra de progreso con bordes redondeados
    bar_x, bar_y = 160, y + 15
    bar_width, bar_height = 200, 20
    cv2.rectangle(panel_fila, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (220, 220, 220), -1)
    progress = np.interp(valor, [0, 180], [0, bar_width])
    cv2.rectangle(panel_fila, (bar_x, bar_y), (int(bar_x + progress), bar_y + bar_height), color, -1)
    cv2.rectangle(panel_fila, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (180, 180, 180), 1)

    # Valor numérico
    cv2.putText(panel_fila, str(int(valor)), (bar_x + bar_width + 15, y + 32), font_valores, 0.7, (0, 0, 0), 2)


def crear_panel_superior(ancho_total, alto=90, logo_img=None):
    """Crea el panel superior con el título y el logo, manteniendo el dorado característico."""
    panel = np.zeros((alto, ancho_total, 3), dtype=np.uint8)
    panel[:] = (224, 161, 0)  # color original (dorado PUCE)
    font = cv2.FONT_HERSHEY_DUPLEX

    texto = "PROYECTO BRAZO ROBÓTICO"
    escala, grosor = 1.2, 2
    texto_size = cv2.getTextSize(texto, font, escala, grosor)[0]
    texto_x, texto_y = (ancho_total - texto_size[0]) // 2, 60

    # Sombra para mejor legibilidad
    cv2.putText(panel, texto, (texto_x + 3, texto_y + 3), font, escala, (0, 0, 0), grosor + 1)
    cv2.putText(panel, texto, (texto_x, texto_y), font, escala, COLOR_TEXTO_CLARO, grosor)

    # Logo institucional (si existe)
    if logo_img is not None:
        try:
            logo_h = alto - 30
            logo_w = int(logo_h * logo_img.shape[1] / logo_img.shape[0])
            logo_resized = cv2.resize(logo_img, (logo_w, logo_h))
            panel[15:15 + logo_h, 30:30 + logo_w] = logo_resized
        except Exception as e:
            print(f"Error dibujando logo: {e}")

    return panel


def crear_panel_lateral(ancho, alto, angulos, conexion_activa, modo_actual,
    servo_en_prueba=None, tiempo_restante=0, postura_activa=None,
    articulacion_seleccionada_idx=-1):
    """Crea el panel lateral con un diseño más limpio y moderno."""
    panel = np.ones((alto, ancho, 3), dtype=np.uint8) * 250
    font_texto = cv2.FONT_HERSHEY_SIMPLEX
    font_modo = cv2.FONT_HERSHEY_DUPLEX

    # Estado de conexión
    estado_conexion = "CONECTADO" if conexion_activa else "DESCONECTADO"
    color_conexion = (0, 180, 0) if conexion_activa else (0, 0, 180)
    cv2.putText(panel, f"CONEXIÓN: {estado_conexion}", (30, 50), font_texto, 0.9, color_conexion, 2)

    # Modo actual
    modos_info = {
        config.MODO_CONFIGURACION: ("MODO CONFIGURACIÓN", (0, 255, 255), (40, 40, 40)),
        config.MODO_POSTURA: ("MODO POSTURA", (255, 255, 255), (128, 0, 128)),
        config.MODO_PRUEBA: ("MODO PRUEBA", (255, 255, 255), (200, 100, 0)),
        config.MODO_PAUSA: ("MODO PAUSA", (255, 255, 255), (0, 165, 255)),
        config.MODO_GESTO_SI: ("MODO GESTO: SÍ", (255, 255, 255), (100, 200, 0)),
        config.MODO_GESTO_NO: ("MODO GESTO: NO", (255, 255, 255), (0, 100, 200)),
        config.MODO_NORMAL: ("MODO OPERACIÓN", (0, 255, 0), (40, 40, 40)),
        config.MODO_MANUAL: ("MODO MANUAL", (255, 255, 0), (0, 70, 120))
    }

    texto_modo, color_texto, color_fondo = modos_info.get(modo_actual, ("MODO DESCONOCIDO", (255, 0, 255), (0, 0, 0)))
    cv2.rectangle(panel, (20, 80), (ancho - 20, 140), color_fondo, -1)
    texto_size = cv2.getTextSize(texto_modo, font_modo, 1.1, 2)[0]
    cv2.putText(panel, texto_modo, ((ancho - texto_size[0]) // 2, 120), font_modo, 1.1, color_texto, 2)

    # Encabezado articulaciones
    cv2.putText(panel, "ESTADO DE ARTICULACIONES", (30, 180), font_texto, 0.8, (0, 0, 0), 2)

    articulaciones = [
        ('proximidad', 'BASE'), ('hombro', 'HOMBRO'), ('codo', 'CODO'),
        ('pitch', 'INCLINACIÓN'), ('roll', 'ROTACIÓN'),
        ('pulgar', 'PULGAR'), ('indice', 'ÍNDICE'), ('medio', 'MEDIO'),
        ('anular', 'ANULAR'), ('menique', 'MEÑIQUE')
    ]

    y_start = 200
    for i, (key, nombre) in enumerate(articulaciones):
        y_pos = y_start + i * 45
        if y_pos + 45 > alto:
            break

        valor = angulos.get(key, 90)
        color = config.COLORES.get(key, COLOR_BARRA_DEFAULT)
        seleccionado = (modo_actual == config.MODO_MANUAL and i == articulacion_seleccionada_idx)

        fila_panel = panel[y_pos:y_pos + 45, :]
        _dibujar_barra_angulo_reducida(fila_panel, 0, nombre, valor, color, seleccionado)

    return panel


def dibujar_zona_calibracion(frame, ancho, alto):
    """Dibuja la zona de calibración semitransparente en el frame."""
    zona_alto = int(alto * 0.20)
    start_y, end_y = (alto - zona_alto) // 2, (alto + zona_alto) // 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, start_y), (ancho, end_y), (0, 255, 255), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    cv2.putText(frame, "ZONA DE CALIBRACIÓN", (ancho // 2 - 220, start_y - 30),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 0, 0), 3)
    cv2.putText(frame, "Mantenga el brazo recto aquí", (ancho // 2 - 220, start_y + 40),
                cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 0, 0), 2)
