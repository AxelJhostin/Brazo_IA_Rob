#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// ============================================
// CONFIGURACIÓN DE HARDWARE
// ============================================
#define SERVOMIN 150  // Pulso mínimo (ajusta según tus servos)
#define SERVOMAX 600  // Pulso máximo (ajusta según tus servos)

// Mapeo de canales PCA9685 a ejes del robot
#define CANAL_BASE 14        // Proximidad/Base
#define CANAL_HOMBRO 12      // Hombro
#define CANAL_CODO 8         // Codo
#define CANAL_PITCH 10       // Pitch (inclinación) - ACTUALMENTE NO USADO
#define CANAL_ROLL 9         // Roll (rotación muñeca)
#define CANAL_PULGAR 13      // Pulgar
#define CANAL_INDICE 5       // Índice
#define CANAL_MEDIO 4        // Medio
#define CANAL_ANULAR 6       // Anular
#define CANAL_MENIQUE 7      // Meñique

// ============================================
// VARIABLES GLOBALES
// ============================================
String inputString = "";      // Buffer para datos entrantes
boolean stringComplete = false;

// Última posición conocida de cada servo
int ultimaPosicion[16] = {90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90};

// ============================================
// FUNCIONES AUXILIARES
// ============================================

// Convierte ángulos (0-180) a pulsos PWM
int gradosAPulso(int angulo) {
  angulo = constrain(angulo, 0, 180);
  return map(angulo, 0, 180, SERVOMIN, SERVOMAX);
}

// Mueve un servo con límites de seguridad
void moverServo(int canal, int angulo) {
  angulo = constrain(angulo, 0, 180);
  
  // Solo envía comando si el ángulo cambió (reduce tráfico I2C)
  if (ultimaPosicion[canal] != angulo) {
    pwm.setPWM(canal, 0, gradosAPulso(angulo));
    ultimaPosicion[canal] = angulo;
  }
}

// Posición de inicio segura (home)
void posicionHome() {
  Serial.println("Moviendo a posición HOME...");
  
  // Articulaciones del brazo
  moverServo(CANAL_BASE, 90);
  moverServo(CANAL_HOMBRO, 90);
  moverServo(CANAL_CODO, 90);
  moverServo(CANAL_PITCH, 90);
  moverServo(CANAL_ROLL, 90);
  
  // Dedos abiertos
  moverServo(CANAL_PULGAR, 20);
  moverServo(CANAL_INDICE, 10);
  moverServo(CANAL_MEDIO, 10);
  moverServo(CANAL_ANULAR, 10);
  moverServo(CANAL_MENIQUE, 10);
  
  delay(500);
  Serial.println("Posición HOME alcanzada");
}

// ============================================
// PROCESAMIENTO DE DATOS DESDE PYTHON
// ============================================

void procesarDatos(String datos) {
  // Formato esperado: <prox,hombro,codo,pitch,roll,pulgar,indice,medio,anular,menique>
  // Ejemplo: <90,120,45,90,90,0,0,0,0,0>
  
  // Remover caracteres < y >
  datos.replace("<", "");
  datos.replace(">", "");
  datos.trim();
  
  // Separar por comas
  int valores[10];
  int index = 0;
  int lastIndex = 0;
  
  for (int i = 0; i < datos.length(); i++) {
    if (datos.charAt(i) == ',' || i == datos.length() - 1) {
      if (i == datos.length() - 1) i++; // Incluir último carácter
      
      String valorStr = datos.substring(lastIndex, i);
      valores[index] = valorStr.toInt();
      index++;
      lastIndex = i + 1;
      
      if (index >= 10) break; // Máximo 10 valores
    }
  }
  
  // Validar que recibimos 10 valores
  if (index == 10) {
    // Aplicar los valores a cada servo
    moverServo(CANAL_BASE, valores[0]);      // Proximidad
    moverServo(CANAL_HOMBRO, valores[1]);    // Hombro
    moverServo(CANAL_CODO, valores[2]);      // Codo
    moverServo(CANAL_PITCH, valores[3]);     // Pitch
    moverServo(CANAL_ROLL, valores[4]);      // Roll
    moverServo(CANAL_PULGAR, valores[5]);    // Pulgar
    moverServo(CANAL_INDICE, valores[6]);    // Índice
    moverServo(CANAL_MEDIO, valores[7]);     // Medio
    moverServo(CANAL_ANULAR, valores[8]);    // Anular
    moverServo(CANAL_MENIQUE, valores[9]);   // Meñique
    
    // Confirmación (opcional, puede ralentizar)
    // Serial.println("OK");
  } else {
    Serial.print("Error: Se esperaban 10 valores, se recibieron ");
    Serial.println(index);
  }
}

// ============================================
// SETUP
// ============================================

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(50); // Timeout corto para respuesta rápida
  
  // Inicializar PCA9685
  pwm.begin();
  pwm.setPWMFreq(50); // Frecuencia para servos (50 Hz)
  
  // Apagar todos los canales inicialmente
  for (int i = 0; i < 16; i++) {
    pwm.setPWM(i, 0, 0);
  }
  
  delay(500);
  
  // Ir a posición HOME
  posicionHome();
  
  Serial.println("=================================");
  Serial.println("BRAZO ROBOTICO - 10 EJES");
  Serial.println("Esperando comandos desde Python...");
  Serial.println("=================================");
  
  inputString.reserve(200); // Reservar memoria para el buffer
}

// ============================================
// LOOP PRINCIPAL
// ============================================

void loop() {
  // Leer datos del puerto serial
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    
    if (inChar == '<') {
      // Inicio de un nuevo comando
      inputString = "";
    } else if (inChar == '>') {
      // Fin del comando, procesar
      procesarDatos(inputString);
      inputString = "";
    } else {
      // Agregar carácter al buffer
      inputString += inChar;
    }
  }
}