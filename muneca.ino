/*
 * CÓDIGO 1: PRUEBA DE GESTOS DE LA MANO (5 DEDOS)
 * Controla solo los servos de los dedos.
 * Mapeo de pines basado en tu PCB.
 */

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// Objeto para el PCA9685
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// Límites de pulso (ajusta si es necesario)
#define SERVOMIN  150
#define SERVOMAX  600

// --- Mapeo de Pines Físicos para la MANO ---
// (Basado en nuestra conversación anterior)
#define PIN_MEÑIQUE    0
#define PIN_ANULAR     1
#define PIN_MEDIO      2
#define PIN_INDICE     3
#define PIN_PULGAR     10

// --- Ángulos para los gestos ---
#define ABIERTO 180
#define CERRADO 0

/**
 * @brief Convierte grados (0-180) a pulsos PWM (SERVOMIN-SERVOMAX)
 */
int gradosAPulso(int angulo) {
  return map(angulo, 0, 180, SERVOMIN, SERVOMAX);
}

/**
 * @brief Mueve un servo específico a un ángulo
 */
void moverDedo(int pinDedo, int angulo) {
  pwm.setPWM(pinDedo, 0, gradosAPulso(angulo));
}

void setup() {
  Serial.begin(115200);
  Serial.println("Prueba de Gestos de la Mano - Iniciando...");

  // Inicia I2C en los pines D2 (SDA) y D1 (SCL) del ESP8266
  Wire.begin(4, 5); // Usamos GPIO 4 (D2) para SDA y GPIO 5 (D1) para SCL
  
  pwm.begin();
  pwm.setPWMFreq(50); // Frecuencia estándar para servos
  
  Serial.println("Mano centrada. Iniciando secuencia...");
  // Posición inicial (medio abiertos)
  moverDedo(PIN_PULGAR, 90);
  moverDedo(PIN_INDICE, 90);
  moverDedo(PIN_MEDIO, 90);
  moverDedo(PIN_ANULAR, 90);
  moverDedo(PIN_MEÑIQUE, 90);
  delay(1000);
}


void loop() {
  // --- 1. SALUDO (ABRIR Y CERRAR) ---
  Serial.println("Gesto: SALUDO (Mano Abierta)");
  moverDedo(PIN_PULGAR, ABIERTO);
  moverDedo(PIN_INDICE, ABIERTO);
  moverDedo(PIN_MEDIO, ABIERTO);
  moverDedo(PIN_ANULAR, ABIERTO);
  moverDedo(PIN_MEÑIQUE, ABIERTO);
  delay(2000);

  Serial.println("Gesto: PUÑO (Mano Cerrada)");
  moverDedo(PIN_PULGAR, CERRADO);
  moverDedo(PIN_INDICE, CERRADO);
  moverDedo(PIN_MEDIO, CERRADO);
  moverDedo(PIN_ANULAR, CERRADO);
  moverDedo(PIN_MEÑIQUE, CERRADO);
  delay(2000);

  // --- 2. GESTO DE ROCK ---
  Serial.println("Gesto: ROCK 🤘");
  moverDedo(PIN_PULGAR, CERRADO);     // Pulgar cerrado
  moverDedo(PIN_INDICE, ABIERTO);     // Índice abierto
  moverDedo(PIN_MEDIO, CERRADO);     // Medio cerrado
  moverDedo(PIN_ANULAR, CERRADO);     // Anular cerrado
  moverDedo(PIN_MEÑIQUE, ABIERTO);    // Meñique abierto
  delay(3000);

  // --- 3. GESTO de PAZ / V ---
  Serial.println("Gesto: PAZ ✌️");
  moverDedo(PIN_PULGAR, CERRADO);     // Pulgar cerrado
  moverDedo(PIN_INDICE, ABIERTO);     // Índice abierto
  moverDedo(PIN_MEDIO, ABIERTO);     // Medio abierto
  moverDedo(PIN_ANULAR, CERRADO);     // Anular cerrado
  moverDedo(PIN_MEÑIQUE, CERRADO);    // Meñique cerrado
  delay(3000);
}