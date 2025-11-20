#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// Crea el objeto para controlar el HW-170 (PCA9685)
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// Límites de pulso para tus servos
#define SERVOMIN  150 // Pulso mínimo (~0 grados)
#define SERVOMAX  600 // Pulso máximo (~180 grados)

// Configuración de prueba
#define TIEMPO_PRUEBA 4000  // 4 segundos por grupo
#define TOTAL_CANALES 16    // El PCA9685 tiene 16 canales (0-15)
#define SERVOS_POR_GRUPO 4  // Probar de 4 en 4

int grupoActual = 0;
unsigned long tiempoAnterior = 0;

int gradosAPulso(int angulo) {
  return map(angulo, 0, 180, SERVOMIN, SERVOMAX);
}

void setup() {
  Serial.begin(115200);
  Serial.println("\n=== TEST COMPLETO PCA9685 ===");
  Serial.println("Probando los 16 canales (0-15)");
  Serial.println("Grupos de 4 servos cada 4 segundos\n");
  
  pwm.begin();
  pwm.setPWMFreq(50);
  
  // Apaga todos los canales inicialmente
  for (int i = 0; i < TOTAL_CANALES; i++) {
    pwm.setPWM(i, 0, 0);
  }
  
  delay(1000);
  tiempoAnterior = millis();
}

void probarGrupo(int grupo) {
  // Calcula qué canales probar en este grupo
  int inicio = grupo * SERVOS_POR_GRUPO;
  int fin = min(inicio + SERVOS_POR_GRUPO, TOTAL_CANALES);
  
  Serial.println("========================================");
  Serial.print("GRUPO ");
  Serial.print(grupo + 1);
  Serial.print(" - Canales: ");
  for (int i = inicio; i < fin; i++) {
    Serial.print(i);
    if (i < fin - 1) Serial.print(", ");
  }
  Serial.println();
  Serial.println("========================================");
  
  // Apaga todos primero
  for (int i = 0; i < TOTAL_CANALES; i++) {
    pwm.setPWM(i, 0, 0);
  }
  
  delay(500);
  
  // Mueve los servos del grupo actual con una secuencia visible
  Serial.println("Moviendo servos...");
  
  // Todos al centro (90°)
  for (int i = inicio; i < fin; i++) {
    pwm.setPWM(i, 0, gradosAPulso(90));
  }
  Serial.println("  -> Posición: 90° (centro)");
  delay(1000);
  
  // Todos a la izquierda (45°)
  for (int i = inicio; i < fin; i++) {
    pwm.setPWM(i, 0, gradosAPulso(45));
  }
  Serial.println("  -> Posición: 45° (izquierda)");
  delay(1000);
  
  // Todos a la derecha (135°)
  for (int i = inicio; i < fin; i++) {
    pwm.setPWM(i, 0, gradosAPulso(135));
  }
  Serial.println("  -> Posición: 135° (derecha)");
  delay(1000);
  
  // Todos al centro nuevamente (90°)
  for (int i = inicio; i < fin; i++) {
    pwm.setPWM(i, 0, gradosAPulso(90));
  }
  Serial.println("  -> Posición: 90° (centro)");
  delay(1000);
  
  Serial.println("\nEsperando para el siguiente grupo...\n");
}

void loop() {
  unsigned long tiempoActual = millis();
  
  // Cada 4 segundos, cambiar al siguiente grupo
  if (tiempoActual - tiempoAnterior >= TIEMPO_PRUEBA) {
    tiempoAnterior = tiempoActual;
    
    // Calcula cuántos grupos hay (16 canales / 4 = 4 grupos)
    int totalGrupos = (TOTAL_CANALES + SERVOS_POR_GRUPO - 1) / SERVOS_POR_GRUPO;
    
    probarGrupo(grupoActual);
    
    grupoActual++;
    
    // Si terminamos todos los grupos, reinicia
    if (grupoActual >= totalGrupos) {
      grupoActual = 0;
      Serial.println("\n");
      Serial.println("*****************************************");
      Serial.println("*   CICLO COMPLETO - TODOS LOS CANALES  *");
      Serial.println("*****************************************");
      Serial.println("\nReiniciando prueba desde el canal 0...\n");
      delay(2000);
    }
  }
}