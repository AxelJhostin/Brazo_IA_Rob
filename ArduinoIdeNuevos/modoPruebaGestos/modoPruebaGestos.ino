#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// Límites de pulso
#define SERVOMIN 150
#define SERVOMAX 600

// Mapear tus servos
#define DEDO1 7
#define DEDO2 5
#define DEDO3 4
#define DEDO4 6
#define PULGAR 13

#define CODO 8
#define ROTACION 9
#define HOMBRO 12
#define BASE 14

// Canal 10 (inclinación) NO SE USA

int gradosAPulso(int angulo) {
  return map(angulo, 0, 180, SERVOMIN, SERVOMAX);
}

void mover(int canal, int angulo) {
  pwm.setPWM(canal, 0, gradosAPulso(angulo));
}

// --------------------------------------------------------
// ------------------ MOVIMIENTOS --------------------------
// --------------------------------------------------------

// Abrir mano (todos los dedos abiertos)
void abrirManoCompleta() {
  mover(DEDO1, 10);
  mover(DEDO2, 10);
  mover(DEDO3, 10);
  mover(DEDO4, 10);
  mover(PULGAR, 20);
  delay(800);
}

// Cerrar mano (puño)
void cerrarManoCompleta() {
  mover(DEDO1, 150);
  mover(DEDO2, 150);
  mover(DEDO3, 150);
  mover(DEDO4, 150);
  mover(PULGAR, 160);
  delay(800);
}

// Pinza (pulgar + dedo 1)
void pinza() {
  mover(DEDO1, 150);
  mover(PULGAR, 150);
  delay(800);
}

void soltarPinza() {
  mover(DEDO1, 10);
  mover(PULGAR, 20);
  delay(800);
}

// Movimiento de saludo
void saludo() {
  for (int i = 0; i < 3; i++) {
    mover(HOMBRO, 60);
    delay(500);
    mover(HOMBRO, 110);
    delay(500);
  }
}

// Rotación izquierda ↔ derecha
void rotarMuneca() {
  mover(ROTACION, 40);
  delay(700);
  mover(ROTACION, 140);
  delay(700);
  mover(ROTACION, 90); // centro
  delay(700);
}

// Mover base izquierda ↔ derecha
void moverBase() {
  mover(BASE, 60);
  delay(800);
  mover(BASE, 120);
  delay(800);
  mover(BASE, 90);
  delay(500);
}

// Mover codo arriba ↔ abajo
void moverCodo() {
  mover(CODO, 40);
  delay(700);
  mover(CODO, 130);
  delay(700);
  mover(CODO, 90);
  delay(700);
}

// Mover hombro adelante ↔ atrás
void moverHombro() {
  mover(HOMBRO, 40);
  delay(700);
  mover(HOMBRO, 120);
  delay(700);
  mover(HOMBRO, 90);
  delay(700);
}

// --------------------------------------------------------
// ----------------- SECUENCIA DEMOSTRACIÓN ----------------
// --------------------------------------------------------

void demo() {

  abrirManoCompleta();
  delay(500);

  cerrarManoCompleta();
  delay(500);

  abrirManoCompleta();
  delay(500);

  pinza();
  delay(700);

  soltarPinza();
  delay(500);

  moverCodo();
  moverHombro();
  delay(300);

  rotarMuneca();
  delay(500);

  moverBase();
  delay(500);

  saludo();
  delay(500);

  // Repite continuamente
}

void setup() {
  Serial.begin(115200);
  Serial.println("Modo demostración iniciado");

  pwm.begin();
  pwm.setPWMFreq(50);

  // Apaga todos al inicio
  for (int i = 0; i < 16; i++) {
    pwm.setPWM(i, 0, 0);
  }

  delay(1000);
}

void loop() {
  demo(); // Repite la demostración
}
