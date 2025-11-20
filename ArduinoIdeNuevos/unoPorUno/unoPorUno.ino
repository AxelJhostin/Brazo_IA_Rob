#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

#define SERVOMIN 150
#define SERVOMAX 600
#define TOTAL_CANALES 16
#define TIEMPO_SERVOMOTOR 8000  // 8 segundos por servo

int gradosAPulso(int angulo) {
  return map(angulo, 0, 180, SERVOMIN, SERVOMAX);
}

void setup() {
  Serial.begin(115200);
  Serial.println("\n=== IDENTIFICACIÓN DE SERVOMOTORES ===");
  Serial.println("Probando todos los canales del PCA9685 (0 a 15)");
  Serial.println("Cada servo se mueve por 8 segundos para identificarlo.\n");

  pwm.begin();
  pwm.setPWMFreq(50);

  // Apaga todo inicialmente
  for (int i = 0; i < TOTAL_CANALES; i++) {
    pwm.setPWM(i, 0, 0);
  }

  delay(1000);
}

void moverServo(int canal) {
  Serial.println("--------------------------------------");
  Serial.print("PROBANDO CANAL: ");
  Serial.println(canal);
  Serial.println("Mira cuál servo se está moviendo...");
  Serial.println("--------------------------------------");

  // Tiempo total: 8 segundos
  // Se divide en 2 segundos por posición

  // 1) Centro (90°) - 2s
  pwm.setPWM(canal, 0, gradosAPulso(90));
  delay(2000);

  // 2) Izquierda (45°) - 2s
  pwm.setPWM(canal, 0, gradosAPulso(45));
  delay(2000);

  // 3) Derecha (135°) - 2s
  pwm.setPWM(canal, 0, gradosAPulso(135));
  delay(2000);

  // 4) Volver al centro (90°) - 2s
  pwm.setPWM(canal, 0, gradosAPulso(90));
  delay(2000);

  // Pausa antes del siguiente
  delay(500);
}

void loop() {
  for (int canal = 0; canal < TOTAL_CANALES; canal++) {

    // Apagar todos antes de probar uno
    for (int i = 0; i < TOTAL_CANALES; i++) {
      pwm.setPWM(i, 0, 0);
    }

    moverServo(canal);
  }

  Serial.println("\n=== CICLO COMPLETO ===");
  Serial.println("Ya se probaron todos los canales.");
  Serial.println("Reiniciando en 3 segundos...\n");
  delay(3000);
}
