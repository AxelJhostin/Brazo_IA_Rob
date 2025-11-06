/*
 * CÓDIGO 2: PRUEBA DE MOVIMIENTO DEL BRAZO (5 EJES)
 * VERSIÓN CORREGIDA (SECUENCIAL)
 * Controla solo los servos de la base, hombro, codo, pitch y roll.
 * Mueve los servos uno por uno para evitar picos de corriente.
 */

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// Objeto para el PCA9685
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// Límites de pulso
#define SERVOMIN  150
#define SERVOMAX  600

// --- Mapeo de Pines Físicos para el BRAZO ---
#define PIN_BASE      9  // Proximidad
#define PIN_HOMBRO    15
#define PIN_CODO      14
#define PIN_PITCH     13 // "Otro codo"
#define PIN_ROLL      11 // Rotación

// --- Límites Seguros (Basados en tu config.py) ---
#define BASE_MIN    30
#define BASE_MAX    150
#define HOMBRO_MIN  30
#define HOMBRO_MAX  160
#define CODO_MIN    20
#define CODO_MAX    170
#define PITCH_MIN   30
#define PITCH_MAX   150
#define ROLL_MIN    10
#define ROLL_MAX    180


/**
 * @brief Convierte grados (0-180) a pulsos PWM (SERVOMIN-SERVOMAX)
 */
int gradosAPulso(int angulo) {
  return map(angulo, 0, 180, SERVOMIN, SERVOMAX);
}

/**
 * @brief Mueve un servo específico a un ángulo
 */
void moverServo(int pinServo, int angulo) {
  pwm.setPWM(pinServo, 0, gradosAPulso(angulo));
}

/**
 * @brief Pone todos los servos del brazo en "home" (90)
 * ¡VERSIÓN CORREGIDA! Mueve los servos secuencialmente.
 */
void posicionHome() {
  Serial.println("Moviendo a HOME (90 grados) secuencialmente...");
  
  // Mueve un servo a la vez con una pequeña pausa
  moverServo(PIN_BASE, 90);
  delay(50); // Pausa corta
  moverServo(PIN_HOMBRO, 90);
  delay(50); // Pausa corta
  moverServo(PIN_CODO, 90);
  delay(50); // Pausa corta
  moverServo(PIN_PITCH, 90);
  delay(50); // Pausa corta
  moverServo(PIN_ROLL, 90);
  
  delay(1000); // Espera a que todos lleguen
}

/**
 * @brief Hace un barrido (sweep) de un servo entre sus límites seguros
 */
void probarServo(int pinServo, int minAng, int maxAng) {
  // Mueve al mínimo
  for (int ang = 90; ang >= minAng; ang--) {
    moverServo(pinServo, ang);
    delay(15);
  }
  delay(500);
  
  // Mueve al máximo
  for (int ang = minAng; ang <= maxAng; ang++) {
    moverServo(pinServo, ang);
    delay(15);
  }
  delay(500);

  // Regresa al centro (90)
  for (int ang = maxAng; ang >= 90; ang--) {
    moverServo(pinServo, ang);
    delay(15);
  }
  delay(500);
}


void setup() {
  Serial.begin(115200);
  Serial.println("Prueba de Movimiento del Brazo (Seguro) - Iniciando...");

  // Inicia I2C en los pines D2 (SDA) y D1 (SCL) del ESP8266
  Wire.begin(4, 5); // Usando GPIO 4 (D2) y GPIO 5 (D1)
  
  pwm.begin();
  pwm.setPWMFreq(50); // Frecuencia estándar para servos
  
  posicionHome(); // Llama a la nueva función 'Home' secuencial
  Serial.println("Inicio de secuencia de prueba.");
}


void loop() {
  Serial.println("--- Probando BASE (Pin 9) ---");
  probarServo(PIN_BASE, BASE_MIN, BASE_MAX);
  delay(1000);

  Serial.println("--- Probando HOMBRO (Pin 15) ---");
  probarServo(PIN_HOMBRO, HOMBRO_MIN, HOMBRO_MAX);
  delay(1000);

  Serial.println("--- Probando CODO (Pin 14) ---");
  probarServo(PIN_CODO, CODO_MIN, CODO_MAX);
  delay(1000);

  Serial.println("--- Probando PITCH (Pin 13) ---");
  probarServo(PIN_PITCH, PITCH_MIN, PITCH_MAX);
  delay(1000);

  Serial.println("--- Probando ROLL (Pin 11) ---");
  probarServo(PIN_ROLL, ROLL_MIN, ROLL_MAX);
  delay(1000);

  Serial.println("Secuencia completada. Reiniciando en 5 segundos...");
  posicionHome(); // Llama a la función 'Home' secuencial de nuevo
  delay(5000);
}