# -*- coding: utf-8 -*-
# =================================================================
# MODULO: ui_components.py (VERSION MEJORADA - DISENO MODERNO)
# DESCRIPCION: Interfaz grafica profesional del controlador del brazo robotico
# MEJORAS: Diseno moderno, iconos visuales, mejor organizacion
# =================================================================

import numpy as np
import cv2
import config

# ========================================
# PALETA DE COLORES MODERNA
# ========================================
# Colores institucionales PUCE
COLOR_CELESTE_PUCE = (255, 200, 100)  # BGR: celeste suave y profesional
COLOR_AZUL_PUCE = (200, 150, 80)      # Azul complementario

# Colores de interfaz moderna
COLOR_FONDO_PRINCIPAL = (245, 245, 245)
COLOR_FONDO_CARDS = (255, 255, 255)
COLOR_TEXTO_PRIMARY = (33, 33, 33)
COLOR_TEXTO_SECONDARY = (117, 117, 117)
COLOR_BORDE_SUAVE = (220, 220, 220)
COLOR_SOMBRA = (200, 200, 200)

# Colores de estado
COLOR_SUCCESS = (76, 175, 80)       # Verde
COLOR_ERROR = (244, 67, 54)         # Rojo
COLOR_WARNING = (255, 152, 0)       # Naranja
COLOR_INFO = (33, 150, 243)         # Azul

# Colores para barras de articulaciones (gradiente profesional)
COLORES_ARTICULACIONES = {
    'proximidad': (66, 133, 244),    # Azul Google
    'hombro': (52, 168, 83),         # Verde
    'codo': (251, 188, 5),           # Amarillo
    'pitch': (234, 67, 53),          # Rojo
    'roll': (156, 39, 176),          # Púrpura
    'pulgar': (255, 87, 34),         # Naranja profundo
    'indice': (255, 193, 7),         # Ámbar
    'medio': (76, 175, 80),          # Verde medio
    'anular': (3, 169, 244),         # Cian
    'menique': (233, 30, 99)         # Rosa
}

# ========================================
# FUNCIONES AUXILIARES DE DIBUJO
# ========================================

def _dibujar_sombra(img, x, y, w, h, intensidad=0.15):
    """Dibuja una sombra sutil para efecto de elevacion."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x+3, y+3), (x+w+3, y+h+3), (0, 0, 0), -1)
    cv2.addWeighted(overlay, intensidad, img, 1-intensidad, 0, img)


def _dibujar_card(img, x, y, w, h, titulo=None, color_borde=COLOR_BORDE_SUAVE):
    """Dibuja una tarjeta (card) con estilo material design."""
    # Sombra
    _dibujar_sombra(img, x, y, w, h, 0.1)
    
    # Fondo de la card
    cv2.rectangle(img, (x, y), (x+w, y+h), COLOR_FONDO_CARDS, -1)
    cv2.rectangle(img, (x, y), (x+w, y+h), color_borde, 2)
    
    # Titulo de la card (opcional)
    if titulo:
        cv2.rectangle(img, (x, y), (x+w, y+35), color_borde, -1)
        cv2.putText(img, titulo, (x+15, y+23), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, COLOR_TEXTO_PRIMARY, 2)


def _dibujar_icono_conexion(img, x, y, conectado=True):
    """Dibuja un icono de estado de conexion."""
    color = COLOR_SUCCESS if conectado else COLOR_ERROR
    # Circulo exterior
    cv2.circle(img, (x, y), 12, color, 2)
    # Circulo interior (relleno)
    cv2.circle(img, (x, y), 7, color, -1)
    
    if conectado:
        # Marca de verificacion
        cv2.line(img, (x-4, y), (x-1, y+3), (255, 255, 255), 2)
        cv2.line(img, (x-1, y+3), (x+5, y-4), (255, 255, 255), 2)
    else:
        # X para desconectado
        cv2.line(img, (x-4, y-4), (x+4, y+4), (255, 255, 255), 2)
        cv2.line(img, (x-4, y+4), (x+4, y-4), (255, 255, 255), 2)


def _dibujar_barra_progreso_moderna(img, x, y, w, h, valor, color, nombre="", mostrar_valor=True):
    """Dibuja una barra de progreso con diseño moderno y minimalista (escala 0-180)."""
    # Fondo de la barra (gris claro)
    cv2.rectangle(img, (x, y), (x+w, y+h), (235, 235, 235), -1)
    cv2.rectangle(img, (x, y), (x+w, y+h), COLOR_BORDE_SUAVE, 1)
    
    # Progreso basado en escala 0-180 grados
    progreso_w = int((valor / 180.0) * w)
    if progreso_w > 0:
        # Barra de progreso principal
        cv2.rectangle(img, (x, y), (x+progreso_w, y+h), color, -1)
        
        # Efecto de brillo en la parte superior
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x+progreso_w, y+h//2), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
    
    # Etiqueta del nombre (izquierda) - sin caracteres especiales problematicos
    if nombre:
        cv2.putText(img, nombre, (x-140, y+h-4), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, COLOR_TEXTO_PRIMARY, 1, cv2.LINE_AA)
    
    # Valor numerico con simbolo de grados (derecha)
    if mostrar_valor:
        # Convertir a entero y agregar simbolo de grados
        valor_texto = f"{int(valor)}"
        cv2.putText(img, valor_texto, (x+w+10, y+h-4), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.55, COLOR_TEXTO_PRIMARY, 2, cv2.LINE_AA)
        # Agregar el circulo del simbolo de grados
        cv2.circle(img, (x+w+10+len(valor_texto)*11+3, y+h-11), 3, COLOR_TEXTO_PRIMARY, 1)
        cv2.circle(img, (x+w+10+len(valor_texto)*11+3, y+h-11), 2, COLOR_TEXTO_PRIMARY, 1)


def _dibujar_indicador_modo(img, x, y, w, h, texto, color_fondo, color_texto):
    """Dibuja un indicador visual del modo actual con estilo moderno."""
    # Sombra
    _dibujar_sombra(img, x, y, w, h, 0.12)
    
    # Fondo con color del modo
    cv2.rectangle(img, (x, y), (x+w, y+h), color_fondo, -1)
    
    # Borde sutil
    cv2.rectangle(img, (x, y), (x+w, y+h), 
                 tuple(max(0, c-30) for c in color_fondo), 2)
    
    # Texto centrado
    font = cv2.FONT_HERSHEY_SIMPLEX  # Fuente más legible
    texto_size = cv2.getTextSize(texto, font, 0.85, 2)[0]
    texto_x = x + (w - texto_size[0]) // 2
    texto_y = y + (h + texto_size[1]) // 2
    
    # Sombra del texto
    cv2.putText(img, texto, (texto_x+2, texto_y+2), font, 0.85, (0, 0, 0), 3, cv2.LINE_AA)
    # Texto principal
    cv2.putText(img, texto, (texto_x, texto_y), font, 0.85, color_texto, 2, cv2.LINE_AA)


# ========================================
# FUNCIÓN PRINCIPAL: PANEL SUPERIOR
# ========================================

def crear_panel_superior(ancho_total, alto=110, logo_img=None):
    """Crea el panel superior con diseno moderno y profesional."""
    panel = np.ones((alto, ancho_total, 3), dtype=np.uint8)
    panel[:] = COLOR_CELESTE_PUCE
    
    # Gradiente sutil (superior a inferior)
    for i in range(alto):
        factor = 1.0 - (i / alto) * 0.15
        panel[i, :] = tuple(int(c * factor) for c in COLOR_CELESTE_PUCE)
    
    # Logo institucional con soporte de transparencia
    if logo_img is not None:
        try:
            logo_h = alto - 25
            logo_w = int(logo_h * logo_img.shape[1] / logo_img.shape[0])
            logo_x, logo_y = 25, 12
            
            # Redimensionar el logo
            logo_resized = cv2.resize(logo_img, (logo_w, logo_h))
            
            # Si el logo tiene canal alpha (transparencia)
            if len(logo_resized.shape) == 3 and logo_resized.shape[2] == 4:
                # Separar los canales BGR y Alpha
                bgr = logo_resized[:, :, :3]
                alpha = logo_resized[:, :, 3:] / 255.0
                
                # Combinar el logo con el fondo usando el canal alpha
                roi = panel[logo_y:logo_y + logo_h, logo_x:logo_x + logo_w]
                blended = (alpha * bgr + (1 - alpha) * roi).astype(np.uint8)
                panel[logo_y:logo_y + logo_h, logo_x:logo_x + logo_w] = blended
            else:
                # Si no tiene transparencia, intentar detectar fondo blanco y hacerlo transparente
                logo_gray = cv2.cvtColor(logo_resized, cv2.COLOR_BGR2GRAY)
                # Crear máscara: píxeles blancos (>240) serán transparentes
                _, mask = cv2.threshold(logo_gray, 240, 255, cv2.THRESH_BINARY_INV)
                mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
                
                # Aplicar la máscara
                roi = panel[logo_y:logo_y + logo_h, logo_x:logo_x + logo_w]
                blended = (mask_3channel * logo_resized + (1 - mask_3channel) * roi).astype(np.uint8)
                panel[logo_y:logo_y + logo_h, logo_x:logo_x + logo_w] = blended
                
        except Exception as e:
            print(f"Error al cargar logo: {e}")
    
    # Titulo principal
    font = cv2.FONT_HERSHEY_SIMPLEX  # Fuente mas legible y normal
    titulo = "CONTROL DE BRAZO ROBOTICO"
    subtitulo = "Sistema de Vision por Computadora"
    
    # Titulo
    escala_titulo, grosor_titulo = 1.4, 2
    texto_size = cv2.getTextSize(titulo, font, escala_titulo, grosor_titulo)[0]
    texto_x = (ancho_total - texto_size[0]) // 2
    
    # Sombra del titulo
    cv2.putText(panel, titulo, (texto_x+3, 50+3), font, escala_titulo, 
               (0, 0, 0), grosor_titulo+1, cv2.LINE_AA)
    # Titulo principal
    cv2.putText(panel, titulo, (texto_x, 50), font, escala_titulo, 
               (255, 255, 255), grosor_titulo, cv2.LINE_AA)
    
    # Subtitulo
    escala_sub, grosor_sub = 0.8, 1
    texto_size_sub = cv2.getTextSize(subtitulo, font, 
                                     escala_sub, grosor_sub)[0]
    texto_x_sub = (ancho_total - texto_size_sub[0]) // 2
    cv2.putText(panel, subtitulo, (texto_x_sub, 80), font, 
               escala_sub, (240, 240, 240), grosor_sub, cv2.LINE_AA)
    
    # Linea decorativa inferior
    cv2.line(panel, (0, alto-3), (ancho_total, alto-3), (255, 255, 255), 3)
    
    return panel


# ========================================
# FUNCIÓN PRINCIPAL: PANEL LATERAL
# ========================================

def crear_panel_lateral(ancho, alto, angulos, conexion_activa, modo_actual,
                       servo_en_prueba=None, tiempo_restante=0, postura_activa=None,
                       articulacion_seleccionada_idx=-1):
    """Crea el panel lateral con diseño moderno tipo dashboard."""
    panel = np.ones((alto, ancho, 3), dtype=np.uint8)
    panel[:] = COLOR_FONDO_PRINCIPAL
    
    y_actual = 20
    margen = 20
    ancho_interno = ancho - 2*margen
    
    # ==========================================
    # SECCION 1: ESTADO DE CONEXION
    # ==========================================
    card_h = 75
    _dibujar_card(panel, margen, y_actual, ancho_interno, card_h)
    
    # Icono de conexion
    _dibujar_icono_conexion(panel, margen+30, y_actual+card_h//2, conexion_activa)
    
    # Texto de conexion
    estado_texto = "CONECTADO" if conexion_activa else "DESCONECTADO"
    color_estado = COLOR_SUCCESS if conexion_activa else COLOR_ERROR
    cv2.putText(panel, "Estado:", (margen+55, y_actual+30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXTO_SECONDARY, 1, cv2.LINE_AA)
    cv2.putText(panel, estado_texto, (margen+55, y_actual+52), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_estado, 2, cv2.LINE_AA)
    
    y_actual += card_h + 15
    
    # ==========================================
    # SECCION 2: MODO DE OPERACION
    # ==========================================
    card_h = 80
    
    modos_info = {
        config.MODO_CONFIGURACION: ("CONFIGURACION", COLOR_WARNING, (33, 33, 33)),
        config.MODO_POSTURA: ("POSTURA", (156, 39, 176), (255, 255, 255)),
        config.MODO_PRUEBA: ("PRUEBA", COLOR_INFO, (255, 255, 255)),
        config.MODO_PAUSA: ("PAUSA", (255, 87, 34), (255, 255, 255)),
        config.MODO_GESTO_SI: ("GESTO: SI", COLOR_SUCCESS, (255, 255, 255)),
        config.MODO_GESTO_NO: ("GESTO: NO", COLOR_ERROR, (255, 255, 255)),
        config.MODO_NORMAL: ("OPERACION", COLOR_SUCCESS, (255, 255, 255)),
        config.MODO_MANUAL: ("MANUAL", (255, 193, 7), (33, 33, 33))
    }
    
    texto_modo, color_fondo, color_texto = modos_info.get(
        modo_actual, ("DESCONOCIDO", (158, 158, 158), (255, 255, 255))
    )
    
    _dibujar_indicador_modo(panel, margen, y_actual, ancho_interno, card_h,
                           f"MODO: {texto_modo}", color_fondo, color_texto)
    
    y_actual += card_h + 15
    
    # ==========================================
    # SECCION 3: ARTICULACIONES DEL BRAZO
    # ==========================================
    _dibujar_card(panel, margen, y_actual, ancho_interno, 280, 
                 "ARTICULACIONES DEL BRAZO", COLOR_CELESTE_PUCE)
    
    y_actual += 50
    
    articulaciones_brazo = [
        ('proximidad', 'BASE'),
        ('hombro', 'HOMBRO'),
        ('codo', 'CODO'),
        ('pitch', 'INCLINACION'),
        ('roll', 'ROTACION')
    ]
    
    for i, (key, nombre) in enumerate(articulaciones_brazo):
        valor = angulos.get(key, 90)
        color = COLORES_ARTICULACIONES.get(key, COLOR_INFO)
        
        # Resaltar si está seleccionado en modo manual
        if modo_actual == config.MODO_MANUAL and i == articulacion_seleccionada_idx:
            cv2.rectangle(panel, (margen+5, y_actual-5), 
                         (ancho-margen-5, y_actual+30), (0, 255, 255), 2)
        
        _dibujar_barra_progreso_moderna(panel, margen+155, y_actual, 180, 18,
                                       valor, color, nombre)
        y_actual += 37
    
    y_actual += 30
    
    # ==========================================
    # SECCION 4: DEDOS DE LA MANO
    # ==========================================
    _dibujar_card(panel, margen, y_actual, ancho_interno, 250,
                 "DEDOS DE LA MANO", (156, 39, 176))
    
    y_actual += 50
    
    dedos = [
        ('pulgar', 'PULGAR'),
        ('indice', 'INDICE'),
        ('medio', 'MEDIO'),
        ('anular', 'ANULAR'),
        ('menique', 'MENIQUE')
    ]
    
    for i, (key, nombre) in enumerate(dedos):
        valor = angulos.get(key, 90)
        color = COLORES_ARTICULACIONES.get(key, COLOR_INFO)
        
        # Resaltar si está seleccionado
        offset_seleccion = len(articulaciones_brazo)
        if modo_actual == config.MODO_MANUAL and (i + offset_seleccion) == articulacion_seleccionada_idx:
            cv2.rectangle(panel, (margen+5, y_actual-5), 
                         (ancho-margen-5, y_actual+30), (0, 255, 255), 2)
        
        _dibujar_barra_progreso_moderna(panel, margen+155, y_actual, 180, 18,
                                       valor, color, nombre)
        y_actual += 37
    
    # ==========================================
    # SECCION 5: INFORMACION ADICIONAL
    # ==========================================
    y_actual = alto - 80
    
    # Instrucciones o información contextual
    info_texto = ""
    if modo_actual == config.MODO_MANUAL:
        info_texto = "Usa flechas para ajustar"
    elif modo_actual == config.MODO_CONFIGURACION:
        info_texto = "Calibrando sistema..."
    elif modo_actual == config.MODO_PAUSA:
        info_texto = "Sistema en pausa"
    elif servo_en_prueba:
        info_texto = f"Probando: {servo_en_prueba.upper()}"
    
    if info_texto:
        cv2.rectangle(panel, (margen, y_actual), (ancho-margen, y_actual+50),
                     COLOR_INFO, -1)
        cv2.rectangle(panel, (margen, y_actual), (ancho-margen, y_actual+50),
                     tuple(max(0, c-30) for c in COLOR_INFO), 2)
        
        texto_size = cv2.getTextSize(info_texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        texto_x = margen + (ancho_interno - texto_size[0]) // 2
        cv2.putText(panel, info_texto, (texto_x, y_actual+30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return panel


# ========================================
# FUNCIÓN: ZONA DE CALIBRACIÓN
# ========================================

def dibujar_zona_calibracion(frame, ancho, alto):
    """Dibuja la zona de calibración con diseño moderno y atractivo."""
    zona_alto = int(alto * 0.25)
    start_y = (alto - zona_alto) // 2
    end_y = start_y + zona_alto
    
    # Fondo semitransparente con gradiente
    overlay = frame.copy()
    for i in range(start_y, end_y):
        alpha = 0.3 * (1 - abs(i - (start_y + zona_alto/2)) / (zona_alto/2))
        color = tuple(int(c * (1-alpha) + 0 * alpha) for c in (0, 255, 255))
        cv2.rectangle(overlay, (0, i), (ancho, i+1), (0, 255, 255), -1)
    
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    
    # Bordes de la zona
    cv2.rectangle(frame, (0, start_y), (ancho, end_y), (0, 255, 255), 4)
    cv2.line(frame, (0, start_y), (ancho, start_y), (0, 200, 200), 6)
    cv2.line(frame, (0, end_y), (ancho, end_y), (0, 200, 200), 6)
    
    # Título con sombra
    font = cv2.FONT_HERSHEY_SIMPLEX  # Fuente más legible
    titulo = "ZONA DE CALIBRACION"
    texto_size = cv2.getTextSize(titulo, font, 1.3, 2)[0]
    texto_x = (ancho - texto_size[0]) // 2
    texto_y = start_y - 25
    
    # Sombra
    cv2.putText(frame, titulo, (texto_x+3, texto_y+3), font, 1.3, (0, 0, 0), 4, cv2.LINE_AA)
    # Texto principal
    cv2.putText(frame, titulo, (texto_x, texto_y), font, 1.3, (0, 255, 255), 2, cv2.LINE_AA)
    
    # Instrucciones
    instruccion = "Mantener el brazo recto dentro de esta zona"
    texto_size_inst = cv2.getTextSize(instruccion, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    texto_x_inst = (ancho - texto_size_inst[0]) // 2
    
    cv2.putText(frame, instruccion, (texto_x_inst+2, start_y+45+2), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, instruccion, (texto_x_inst, start_y+45), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Indicadores visuales (flechas apuntando al centro)
    centro_y = (start_y + end_y) // 2
    # Flecha izquierda
    cv2.arrowedLine(frame, (50, centro_y), (150, centro_y), (0, 255, 255), 4, tipLength=0.3)
    # Flecha derecha
    cv2.arrowedLine(frame, (ancho-50, centro_y), (ancho-150, centro_y), (0, 255, 255), 4, tipLength=0.3)