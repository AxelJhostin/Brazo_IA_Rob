# =================================================================
# MÓDULO: ui_components.py
# DESCRIPCIÓN: Contiene todas las funciones para dibujar la
#               interfaz gráfica del controlador del brazo robótico.
# VERSIÓN: 2.1 - Añadido resaltado para Modo Manual.
# =================================================================

import numpy as np
import cv2
import config

def _dibujar_barra_angulo(panel, y, nombre, valor, color, seleccionado=False):
    """Dibuja una etiqueta, una barra de progreso y el valor numérico para un ángulo."""
    font_texto = cv2.FONT_HERSHEY_SIMPLEX
    font_valores = cv2.FONT_HERSHEY_DUPLEX
    ancho_panel = panel.shape[1]

    # Resalta la fila si está seleccionada
    if seleccionado:
        cv2.rectangle(panel, (10, y + 5), (ancho_panel - 10, y + 60), (0, 255, 255, 0.5), -1)
        cv2.rectangle(panel, (10, y + 5), (ancho_panel - 10, y + 60), (0, 200, 200), 2)

    cv2.putText(panel, nombre, (30, y + 38), font_texto, 0.8, (60, 60, 60), 2)
    
    bar_x = 180; bar_width = 200
    cv2.rectangle(panel, (bar_x, y + 20), (bar_x + bar_width, y + 45), (220, 220, 220), -1)
    
    progress = np.interp(valor, [0, 180], [0, bar_width])
    cv2.rectangle(panel, (bar_x, y + 20), (int(bar_x + progress), y + 45), color, -1)
    
    cv2.putText(panel, str(int(valor)), (bar_x + bar_width + 15, y + 42), font_valores, 0.9, (0, 0, 0), 2)


def crear_panel_superior(ancho_total, alto=90, logo_img=None):
    # ... (sin cambios)
    panel = np.zeros((alto, ancho_total, 3), dtype=np.uint8)
    panel[:] = (224, 161, 0)
    font = cv2.FONT_HERSHEY_DUPLEX
    grosor, escala = 2, 1.2
    texto = "PROYECTO BRAZO ROBOTICO"
    texto_size = cv2.getTextSize(texto, font, escala, grosor)[0]
    texto_x, texto_y = (ancho_total - texto_size[0]) // 2, 60
    cv2.putText(panel, texto, (texto_x + 2, texto_y + 2), font, escala, (0, 0, 0), grosor + 1)
    cv2.putText(panel, texto, (texto_x, texto_y), font, escala, (255, 255, 255), grosor)
    if logo_img is not None:
        try:
            logo_h = alto - 30
            logo_w = int(logo_h * logo_img.shape[1] / logo_img.shape[0])
            logo_resized = cv2.resize(logo_img, (logo_w, logo_h))
            logo_bg = np.ones((logo_h, logo_w, 3), dtype=np.uint8) * 255
            logo_bg[0:logo_h, 0:logo_w] = logo_resized
            panel[15:15 + logo_h, 30:30 + logo_w] = logo_bg
        except Exception as e: print(f"Error dibujando logo: {str(e)}")
    return panel


def crear_panel_lateral(ancho, alto, angulos, mano_estable, conexion_activa, modo_actual, servo_en_prueba=None, tiempo_restante=0, postura_activa=None, articulacion_seleccionada_idx=-1):
    panel = np.ones((alto, ancho, 3), dtype=np.uint8) * 250
    font_modo, font_texto = cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_SIMPLEX
    grosor_normal, grosor_modo = 2, 2
    
    estado_conexion = "CONECTADO" if conexion_activa else "DESCONECTADO"
    color_conexion = (0, 180, 0) if conexion_activa else (0, 0, 180)
    cv2.putText(panel, f"CONEXIÓN: {estado_conexion}", (30, 50), font_texto, 0.9, color_conexion, grosor_normal)

    modos_info = {
        config.MODO_CONFIGURACION: ("MODO CONFIGURACION", (0, 255, 255), (40, 40, 40)),
        config.MODO_POSTURA: ("MODO POSTURA", (255, 255, 255), (128, 0, 128)),
        config.MODO_PRUEBA: ("MODO PRUEBA", (255, 255, 255), (200, 100, 0)),
        config.MODO_PAUSA: ("MODO PAUSA", (255, 255, 255), (0, 165, 255)),
        config.MODO_GESTO_SI: ("MODO GESTO: SI", (255, 255, 255), (100, 200, 0)),
        config.MODO_GESTO_NO: ("MODO GESTO: NO", (255, 255, 255), (0, 100, 200)),
        config.MODO_NORMAL: ("MODO OPERACION", (0, 255, 0), (40, 40, 40)),
        config.MODO_MANUAL: ("MODO MANUAL", (255, 255, 0), (0, 70, 120))
    }
    
    info = modos_info.get(modo_actual, ("MODO DESCONOCIDO", (255, 0, 255), (0,0,0)))
    texto_modo, color_texto, color_fondo = info
    
    cv2.rectangle(panel, (20, 80), (ancho - 20, 140), color_fondo, -1)
    texto_size = cv2.getTextSize(texto_modo, font_modo, 1.1, grosor_modo)[0]
    cv2.putText(panel, texto_modo, ((ancho - texto_size[0]) // 2, 120), font_modo, 1.1, color_texto, grosor_modo)
    
    cv2.putText(panel, "ESTADO DE ARTICULACIONES", (30, 190), font_texto, 1, (0, 0, 0), 2)
    y_start = 220
    
    articulaciones = [('proximidad', 'BASE'), ('hombro', 'HOMBRO'), ('codo', 'CODO'), 
                      ('pitch', 'INCLINACION'), ('roll', 'ROTACION')]
    color_barra_azul = (255,191,0) # Azul claro en formato BGR
    
    for i, (key, nombre) in enumerate(articulaciones):
        y_pos = y_start + i * 65
        valor = angulos.get(key, 90)
        color = config.COLORES.get(key, (0, 0, 0))
        seleccionado = (modo_actual == config.MODO_MANUAL and i == articulacion_seleccionada_idx)
        _dibujar_barra_angulo(panel, y_pos, nombre, valor, color_barra_azul, seleccionado)

    y_pinza = y_start + len(articulaciones) * 65
    seleccionado_pinza = (modo_actual == config.MODO_MANUAL and articulacion_seleccionada_idx == len(articulaciones))
    if seleccionado_pinza:
        cv2.rectangle(panel, (10, y_pinza + 5), (ancho - 10, y_pinza + 60), (0, 255, 255, 0.5), -1)
        cv2.rectangle(panel, (10, y_pinza + 5), (ancho - 10, y_pinza + 60), (0, 200, 200), 2)

    cv2.putText(panel, "PINZA", (30, y_pinza + 38), font_texto, 0.8, (60, 60, 60), 2)
    estado_pinza = "CERRADA" if angulos.get('mano', 0) == 1 else "ABIERTA"
    color_pinza = (200, 50, 50) if angulos.get('mano', 0) == 1 else (50, 200, 50)
    cv2.rectangle(panel, (180, y_pinza + 20), (180 + 200, y_pinza + 45), color_pinza, -1)
    cv2.putText(panel, estado_pinza, (225, y_pinza + 42), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 2)

    return panel

def dibujar_zona_calibracion(frame, ancho, alto):
    # ... (sin cambios)
    zona_alto = int(alto * 0.20)
    start_y, end_y = (alto - zona_alto) // 2, (alto + zona_alto) // 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, start_y), (ancho, end_y), (0, 255, 255), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    cv2.putText(frame, "ZONA DE CALIBRACION", (ancho // 2 - 220, start_y - 30), cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 0, 0), 3)
    cv2.putText(frame, "Mantenga el brazo recto aqui", (ancho // 2 - 220, start_y + 40), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 0, 0), 2)
