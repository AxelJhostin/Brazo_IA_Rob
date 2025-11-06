/*
 * CÓDIGO 1: PRUEBA DE GESTOS DE LA MANO (5 DEDOS)
 * VERSIÓN CORREGIDA (SECUENCIAL)
 * Mueve los dedos uno por uno para evitar picos de corriente.
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

// Pausa corta entre movimientos de dedos (en milisegundos)
#define PAUSA_DEDO 50 

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
  Serial.println("Prueba de Gestos de la Mano (Segura) - Iniciando...");

  // Inicia I2C en los pines D2 (SDA) y D1 (SCL) del ESP8266
  // Usamos GPIO 4 (D2) y GPIO 5 (D1)
  Wire.begin(4, 5); 
  
  pwm.begin();
  pwm.setPWMFreq(50); // Frecuencia estándar para servos
  
  Serial.println("Mano centrada. Iniciando secuencia...");
  // Posición inicial (medio abiertos) - ¡AHORA SECUENCIAL!
  moverDedo(PIN_PULGAR, 90);
  delay(PAUSA_DEDO);
  moverDedo(PIN_INDICE, 90);
  delay(PAUSA_DEDO);
  moverDedo(PIN_MEDIO, 90);
  delay(PAUSA_DEDO);
  moverDedo(PIN_ANULAR, 90);
  delay(PAUSA_DEDO);
  moverDedo(PIN_MEÑIQUE, 90);
  delay(1000);
}


void loop() {
  // --- 1. SALUDO (ABRIR) ---
  Serial.println("Gesto: SALUDO (Mano Abierta)");
  moverDedo(PIN_PULGAR, ABIERTO);
  delay(PAUSA_DEDO);
  moverDedo(PIN_INDICE, ABIERTO);
  delay(PAUSA_DEDO);
  moverDedo(PIN_MEDIO, ABIERTO);
  delay(PAUSA_DEDO);
  moverDedo(PIN_ANULAR, ABIERTO);
  delay(PAUSA_DEDO);
  moverDedo(PIN_MEÑIQUE, ABIERTO);
  delay(2000); // Pausa larga para ver el gesto

  // --- 2. PUÑO (CERRAR) ---
  Serial.println("Gesto: PUÑO (Mano Cerrada)");
  moverDedo(PIN_PULGAR, CERRADO);
  delay(PAUSA_DEDO);
  moverDedo(PIN_INDICE, CERRADO);
  delay(PAUSA_DEDO);
  moverDedo(PIN_MEDIO, CERRADO);
  delay(PAUSA_DEDO);
  moverDedo(PIN_ANULAR, CERRADO);
  delay(PAUSA_DEDO);
  moverDedo(PIN_MEÑIQUE, CERRADO);
  delay(2000);

  // --- 3. GESTO DE ROCK ---
  Serial.println("Gesto: ROCK 🤘");
  moverDedo(PIN_PULGAR, CERRADO);
  delay(PAUSA_DEDO);
  moverDedo(PIN_INDICE, ABIERTO);
  delay(PAUSA_DEDO);
  moverDedo(PIN_MEDIO, CERRADO);
  delay(PAUSA_DEDO);
  moverDedo(PIN_ANULAR, CERRADO);
  delay(PAUSA_DEDO);
  moverDedo(PIN_MEÑIQUE, ABIERTO);
  delay(3000);

  // --- 4. GESTO de PAZ / V ---
  Serial.println("Gesto: PAZ ✌️");
  moverDedo(PIN_PULGAR, CERRADO);
  delay(PAUSA_DEDO);
  moverDedo(PIN_INDICE, ABIERTO);
  delay(PAUSA_DEDO);
  moverDedo(PIN_MEDIO, ABIERTO);
  delay(PAUSA_DEDO);
  moverDedo(PIN_ANULAR, CERRADO);
  delay(PAUSA_DEDO);
  moverDedo(PIN_MEÑIQUE, CERRADO);
  delay(3000);

  // --- 5. ¡NUEVO! GESTO DE OK ---
  Serial.println("Gesto: OK 👌");
  moverDedo(PIN_PULGAR, CERRADO); // O puedes probar 90
  delay(PAUSA_DEDO);
  moverDedo(PIN_INDICE, CERRADO); // O puedes probar 90
  delay(PAUSA_DEDO);
  moverDedo(PIN_MEDIO, ABIERTO);
  delay(PAUSA_DEDO);
  moverDedo(PIN_ANULAR, ABIERTO);
  delay(PAUSA_DEDO);
  moverDedo(PIN_MEÑIQUE, ABIERTO);
  delay(3000);

  // --- 6. ¡NUEVO! GESTO DE PULGAR ARRIBA ---
  Serial.println("Gesto: PULGAR ARRIBA 👍");
  moverDedo(PIN_PULGAR, ABIERTO);
  delay(PAUSA_DEDO);
  moverDedo(PIN_INDICE, CERRADO);
  delay(PAUSA_DEDO);
  moverDedo(PIN_MEDIO, CERRADO);
  delay(PAUSA_DEDO);
  moverDedo(PIN_ANULAR, CERRADO);
  delay(PAUSA_DEDO);
  moverDedo(PIN_MEÑIQUE, CERRADO);
  delay(3000);
}