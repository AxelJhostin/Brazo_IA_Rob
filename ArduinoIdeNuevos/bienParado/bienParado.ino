#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

// ====== CONFIG ======
#define SERVOMIN 150
#define SERVOMAX 600

// Servos fijos (se mantienen en "90" + offset)
int servosFijos[] = {0, 1, 2, 3, 4, 9, 10, 11, 13, 14, 15};
// Dedos (animación coordinada)
int dedos[] = {5, 6, 7, 8, 12};

// Offsets por canal (en grados) -> ajustar con calibración si un servo considera 90° distinto
// índice = canal, valor = desplazamiento (ej: si canal 4 necesita +5 grados para quedar "recto", poner +5)
int offsets[16] = {0,0,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

// Límite de ángulos permitidos para seguridad (evita mandar pulsos fuera de rango)
const int ANG_MIN = 0;
const int ANG_MAX = 180;

// ====== UTILS ======
int gradosAPulso(int angulo) {
  angulo = constrain(angulo, ANG_MIN, ANG_MAX);
  return map(angulo, 0, 180, SERVOMIN, SERVOMAX);
}

void setServoAnguloRaw(int canal, int angulo) {
  pwm.setPWM(canal, 0, gradosAPulso(angulo));
}

// Mueve un servo aplicando su offset
void mover(int canal, int angulo) {
  if (canal < 0 || canal > 15) return;
  int target = angulo + offsets[canal];
  target = constrain(target, ANG_MIN, ANG_MAX);
  setServoAnguloRaw(canal, target);
}

// Refuerza la postura (manda 90+offset a todos los servos fijos)
void mantenerServosFijos() {
  for (unsigned i = 0; i < sizeof(servosFijos)/sizeof(servosFijos[0]); i++) {
    mover(servosFijos[i], 90);
  }
}

// Interpolador (easing lineal simple) entre angA -> angB en pasos
void moverSuave(int canal, int angA, int angB, int pasos, int pausaMs) {
  for (int s = 0; s <= pasos; s++) {
    float t = (float)s / (float)pasos;
    int ang = round(angA + (angB - angA) * t);
    mover(canal, ang);
    // refuerza servos fijos cada iteración (reduce vibraciones)
    mantenerServosFijos();
    delay(pausaMs);
  }
}

// Animación coordinada de dedos con easing
void animacionDedosCoordinada(int repeticiones = 1) {
  const int pasos = 30;      // más pasos = más suave
  const int pausa = 15;      // ms entre pasos (ajusta velocidad)
  for (int r = 0; r < repeticiones; r++) {
    // cerrar mano (0 -> 90)
    for (int i = 0; i < (int)(sizeof(dedos)/sizeof(dedos[0])); i++) {
      // para un efecto coordinado, aplicamos pequeñas fases (delay entre dedos)
      int canal = dedos[i];
      // mover suavemente de 0 a 90
      moverSuave(canal, 0, 90, pasos, pausa);
      // pequeña pausa entre dedos para sensación escalonada
      delay(40);
    }
    delay(300);

    // abrir mano (90 -> 0) inverso para coordinación
    for (int i = (int)(sizeof(dedos)/sizeof(dedos[0])) - 1; i >= 0; i--) {
      int canal = dedos[i];
      moverSuave(canal, 90, 0, pasos, pausa);
      delay(40);
    }
    delay(300);
  }
}

// Configuración inicial: posiciona todos en 90 (con offsets) y dedos en 0 (abiertos)
void setPosturaExposicion() {
  // dedos abiertos
  for (unsigned i = 0; i < sizeof(dedos)/sizeof(dedos[0]); i++) {
    mover(dedos[i], 0);
  }
  // servos fijos a 90
  mantenerServosFijos();
}

void setup() {
  Serial.begin(115200);
  Serial.println("Iniciando postura de exposicion con dedos animados...");

  pwm.begin();
  pwm.setPWMFreq(50);

  // Apaga/establece servos
  for (int c = 0; c < 16; c++) {
    pwm.setPWM(c, 0, 0);
  }
  delay(300);

  setPosturaExposicion();
  delay(300);
}

void loop() {
  // Mantén la postura luego anima dedos. MantenerServosFijos se llama continuamente durante la animación.
  animacionDedosCoordinada(1);
  // Puedes añadir pauses largas entre ciclos si quieres:
  delay(200);
}
