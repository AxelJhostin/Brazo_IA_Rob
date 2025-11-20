#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

#define SERVOMIN 150
#define SERVOMAX 600

#define CANAL_BASE 14
#define CANAL_HOMBRO 12
#define CANAL_CODO 8
#define CANAL_PITCH 10
#define CANAL_ROLL 9
#define CANAL_PULGAR 13
#define CANAL_INDICE 6
#define CANAL_MEDIO 4
#define CANAL_ANULAR 5
#define CANAL_MENIQUE 7

String inputString = "";
boolean stringComplete = false;

int ultimaPosicion[16] = {90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90};

int gradosAPulso(int angulo) {
  angulo = constrain(angulo, 0, 180);
  return map(angulo, 0, 180, SERVOMIN, SERVOMAX);
}

void moverServo(int canal, int angulo) {
  angulo = constrain(angulo, 0, 180);
  
  if (ultimaPosicion[canal] != angulo) {
    pwm.setPWM(canal, 0, gradosAPulso(angulo));
    ultimaPosicion[canal] = angulo;
  }
}

void posicionHome() {
  Serial.println("Moviendo a posicion HOME...");
  
  moverServo(CANAL_BASE, 90);
  moverServo(CANAL_HOMBRO, 90);
  moverServo(CANAL_CODO, 90);
  moverServo(CANAL_PITCH, 90);
  moverServo(CANAL_ROLL, 90);
  
  moverServo(CANAL_PULGAR, 20);
  moverServo(CANAL_INDICE, 10);
  moverServo(CANAL_MEDIO, 10);
  moverServo(CANAL_ANULAR, 10);
  moverServo(CANAL_MENIQUE, 10);
  
  delay(500);
  Serial.println("Posicion HOME alcanzada");
}

void procesarDatos(String datos) {
  datos.replace("<", "");
  datos.replace(">", "");
  datos.trim();
  
  int valores[10];
  int index = 0;
  int lastIndex = 0;
  
  for (int i = 0; i < datos.length(); i++) {
    if (datos.charAt(i) == ',' || i == datos.length() - 1) {
      if (i == datos.length() - 1) i++;
      
      String valorStr = datos.substring(lastIndex, i);
      valores[index] = valorStr.toInt();
      index++;
      lastIndex = i + 1;
      
      if (index >= 10) break;
    }
  }
  
  if (index == 10) {
    moverServo(CANAL_BASE, valores[0]);
    moverServo(CANAL_HOMBRO, valores[1]);
    moverServo(CANAL_CODO, valores[2]);
    moverServo(CANAL_PITCH, valores[3]);
    moverServo(CANAL_ROLL, valores[4]);
    moverServo(CANAL_PULGAR, valores[5]);
    moverServo(CANAL_INDICE, valores[6]);
    moverServo(CANAL_MEDIO, valores[7]);
    moverServo(CANAL_ANULAR, valores[8]);
    moverServo(CANAL_MENIQUE, valores[9]);
  } else {
    Serial.print("Error: Se esperaban 10 valores, se recibieron ");
    Serial.println(index);
  }
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(50);
  
  pwm.begin();
  pwm.setPWMFreq(50);
  
  for (int i = 0; i < 16; i++) {
    pwm.setPWM(i, 0, 0);
  }
  
  delay(500);
  
  posicionHome();
  
  Serial.println("=================================");
  Serial.println("BRAZO ROBOTICO - 10 EJES - COM5");
  Serial.println("Esperando comandos desde Python...");
  Serial.println("=================================");
  
  inputString.reserve(200);
}

void loop() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    
    if (inChar == '<') {
      inputString = "";
    } else if (inChar == '>') {
      procesarDatos(inputString);
      inputString = "";
    } else {
      inputString += inChar;
    }
  }
}
