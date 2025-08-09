# =================================================================
# MÓDULO: ui_components.py
# DESCRIPCIÓN: Contiene todas las funciones para dibujar la
#               interfaz gráfica del controlador del brazo robótico.
# VERSIÓN: 1.8 - Muestra la postura activa en la GUI
# =================================================================

import numpy as np
import cv2
import config

def crear_panel_superior(ancho_total, alto=90, logo_img=None):
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

def crear_panel_lateral(ancho, alto, angulos, mano_estable, conexion_activa, modo_actual, servo_en_prueba=None, tiempo_restante=0, postura_activa=None):
    panel = np.ones((alto, ancho, 3), dtype=np.uint8) * 255
    font_modo, font_texto, font_valores = cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_SIMPLEX
    grosor_normal, grosor_modo, grosor_valores = 2, 2, 2
    
    estado_conexion = "CONECTADO" if conexion_activa else "DESCONECTADO"
    color_conexion = (0, 180, 0) if conexion_activa else (0, 0, 180)
    cv2.putText(panel, f"CONEXIÓN: {estado_conexion}", (30, 50), font_texto, 0.9, color_conexion, grosor_normal)

    y_start = 160
    if modo_actual == config.MODO_CONFIGURACION:
        cv2.rectangle(panel, (20, 70), (ancho - 20, 130), (40, 40, 40), -1)
        cv2.putText(panel, "MODO CONFIGURACION", (ancho // 2 - 220, 110), font_modo, 1.1, (0, 255, 255), grosor_modo)
        if tiempo_restante > 0:
            cv2.rectangle(panel, (20, 140), (ancho - 20, 180), (240, 240, 200), -1)
            cv2.putText(panel, f"CALIBRANDO: {tiempo_restante:.1f}s", (ancho//2-150, 170), font_texto, 0.9, (0,0,0), grosor_normal)
        else:
            cv2.rectangle(panel, (20, 140), (ancho - 20, 180), (240, 240, 200), -1)
            cv2.putText(panel, "COLOCAR BRAZO EN ZONA AMARILLA", (ancho//2-250, 170), font_texto, 0.8, (0,0,0), grosor_normal)
        cv2.line(panel, (25, 190), (ancho - 25, 190), (200, 200, 200), 2)
        y_start = 220
    elif modo_actual == config.MODO_POSTURA:
        cv2.rectangle(panel, (20, 70), (ancho - 20, 130), (128, 0, 128), -1)
        cv2.putText(panel, "MODO POSTURA", (ancho // 2 - 150, 110), font_modo, 1.1, (255, 255, 255), grosor_modo)
        # ¡MEJORA! Muestra qué postura está activa.
        if postura_activa:
            cv2.putText(panel, f"Activa: {postura_activa.upper()}", (ancho // 2 - 100, 170), font_texto, 0.9, (0,0,0), grosor_normal)
        cv2.line(panel, (25, 190), (ancho - 25, 190), (200, 200, 200), 2)
        y_start = 220
    elif modo_actual == config.MODO_PRUEBA:
        cv2.rectangle(panel, (20, 70), (ancho - 20, 130), (200, 100, 0), -1)
        cv2.putText(panel, "MODO PRUEBA", (ancho // 2 - 140, 110), font_modo, 1.1, (255, 255, 255), grosor_modo)
        if servo_en_prueba:
            texto_prueba = "TODOS LOS SERVOS" if servo_en_prueba == 'all' else servo_en_prueba.upper()
            cv2.putText(panel, f"Probando: {texto_prueba}", (ancho // 2 - 140, 170), font_texto, 0.9, (0,0,0), grosor_normal)
        cv2.line(panel, (25, 190), (ancho - 25, 190), (200, 200, 200), 2)
        y_start = 220
    elif modo_actual == config.MODO_PAUSA:
        cv2.rectangle(panel, (20, 70), (ancho - 20, 130), (0, 165, 255), -1) # Naranja
        cv2.putText(panel, "MODO PAUSA", (ancho // 2 - 130, 110), font_modo, 1.1, (255, 255, 255), grosor_modo)
        cv2.line(panel, (25, 140), (ancho - 25, 140), (200, 200, 200), 2)
    elif modo_actual == config.MODO_GESTO_SI:
        cv2.rectangle(panel, (20, 70), (ancho - 20, 130), (100, 200, 0), -1)
        cv2.putText(panel, "MODO GESTO: SI", (ancho // 2 - 160, 110), font_modo, 1.1, (255, 255, 255), grosor_modo)
        cv2.line(panel, (25, 140), (ancho - 25, 140), (200, 200, 200), 2)
    elif modo_actual == config.MODO_GESTO_NO:
        cv2.rectangle(panel, (20, 70), (ancho - 20, 130), (0, 100, 200), -1)
        cv2.putText(panel, "MODO GESTO: NO", (ancho // 2 - 160, 110), font_modo, 1.1, (255, 255, 255), grosor_modo)
        cv2.line(panel, (25, 140), (ancho - 25, 140), (200, 200, 200), 2)
    else: # MODO_NORMAL
        cv2.rectangle(panel, (20, 70), (ancho - 20, 130), (40, 40, 40), -1)
        cv2.putText(panel, "MODO OPERACION", (ancho // 2 - 150, 110), font_modo, 1.1, (0, 255, 0), grosor_modo)
        cv2.line(panel, (25, 140), (ancho - 25, 140), (200, 200, 200), 2)

    cv2.putText(panel, "ESTADO DE ARTICULACIONES", (ancho // 2 - 200, y_start + 40), font_texto, 0.9, (60, 60, 60), grosor_normal)
    y_start += 80
    articulaciones = [('proximidad', 'BASE'), ('hombro', 'HOMBRO'), ('codo', 'CODO'), ('pitch', 'INCLINACION'), ('roll', 'ROTACION'), ('mano', 'PINZA')]
    
    for i, (key, nombre) in enumerate(articulaciones):
        y_pos = y_start + i * 60
        color = config.COLORES.get(key, (0, 0, 0))
        cv2.circle(panel, (40, y_pos), 12, color, -1)
        if modo_actual == config.MODO_CONFIGURACION and key != 'mano':
            valor = "-"
        else:
            valor_num = angulos.get(key)
            if key == 'mano':
                valor = "CERRADA" if mano_estable else "ABIERTA"
            elif valor_num is None:
                valor = "N/A"
            else:
                valor = str(int(valor_num))
        cv2.putText(panel, nombre, (80, y_pos + 10), font_texto, 0.9, (0, 0, 0), grosor_normal)
        cv2.putText(panel, f"{valor}", (300, y_pos + 10), font_valores, 0.9, (0, 0, 0), grosor_valores)
    return panel

def dibujar_zona_calibracion(frame, ancho, alto):
    zona_alto = int(alto * 0.20)
    start_y, end_y = (alto - zona_alto) // 2, (alto + zona_alto) // 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, start_y), (ancho, end_y), (0, 255, 255), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    cv2.putText(frame, "ZONA DE CALIBRACION", (ancho // 2 - 220, start_y - 30), cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 0, 0), 3)
    cv2.putText(frame, "Mantenga el brazo recto aqui", (ancho // 2 - 220, start_y + 40), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 0, 0), 2)
    cv2.putText(frame, "para calibrar la distancia", (ancho // 2 - 200, start_y + 80), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 0, 0), 2)