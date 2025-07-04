/*
-----------------------------------------------------------------
 PROYECTO: Receptor para Brazo Robótico de 6 Ejes
 VERSIÓN:  Final Estable (Control por Ángulos 0-180)
-----------------------------------------------------------------
*/

#include <Servo.h>

// Crear un objeto Servo para cada articulación
Servo servoBase;
Servo servoHombro;
Servo servoCodo;
Servo servoMunecaPitch;
Servo servoMunecaRoll;
Servo servoPinza;

// Definir los pines de conexión
const int PIN_BASE = 5;          // D1
const int PIN_HOMBRO = 4;          // D2
const int PIN_CODO = 0;              // D3
const int PIN_MUNECA_PITCH = 14;     // D5
const int PIN_MUNECA_ROLL = 12;      // D6
const int PIN_PINZA = 2;             // D4

void setup() {
  Serial.begin(9600);

  // Conectar cada objeto Servo a su pin.
  // Usamos el attach() simple, sin especificar rango de pulso.
  servoBase.attach(PIN_BASE);
  servoHombro.attach(PIN_HOMBRO);
  servoCodo.attach(PIN_CODO);
  servoMunecaPitch.attach(PIN_MUNECA_PITCH);
  servoMunecaRoll.attach(PIN_MUNECA_ROLL);
  servoPinza.attach(PIN_PINZA);

  // Mover todos los servos a una posición inicial neutra (90 grados)
  servoBase.write(90);
  servoHombro.write(90);
  servoCodo.write(90);
  servoMunecaPitch.write(90);
  servoMunecaRoll.write(90);
  servoPinza.write(90); // Pinza abierta
}

void loop() {
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n');
    if (data.length() > 0) {
      procesarDatos(data);
    }
  }
}

void procesarDatos(String data) {
  int startIndex = data.indexOf('<');
  int endIndex = data.indexOf('>');

  if (startIndex != -1 && endIndex != -1) {
    String values = data.substring(startIndex + 1, endIndex);
    
    int b, h, c, mp, mr, p;
    if (sscanf(values.c_str(), "%d,%d,%d,%d,%d,%d", &b, &h, &c, &mp, &mr, &p) == 6) {
      
      // Convertir el estado de la pinza (0 o 1) a un ángulo
      int anguloPinza = (p == 1) ? 10 : 90; // 1=Cerrada, 0=Abierta

      // Mover los servos a sus nuevas posiciones usando write().
      // Este método es más estable y seguro.
      servoBase.write(b);
      servoHombro.write(h);
      servoCodo.write(c);
      servoMunecaPitch.write(mp);
      servoMunecaRoll.write(mr);
      servoPinza.write(anguloPinza);
    }
  }
}