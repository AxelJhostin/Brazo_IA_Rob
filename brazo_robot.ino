//codigo que se utiliza para el brazo robótico con wi-fi
// Este código es para el ESP8266 y utiliza la librería Servo para controlar los serv
#include <ESP8266WiFi.h>
#include <WiFiUdp.h>
#include <Servo.h>

// =================================================================
// CÓDIGO DEFINITIVO CON WI-FI (UDP)
// =================================================================

// 1. Introduce los datos de tu red Wi-Fi
const char* ssid = "GABIOTA visualnet";
const char* password = "Paloma0320";

// 2. Pines para los 6 servos (los que ya probaste)
// ¡CORRECCIÓN FINAL! Se han intercambiado los pines de la muñeca (Pitch y Roll).
#define PIN_SERVO_1 5  // Base (D1)
#define PIN_SERVO_2 12 // Hombro (D6)
#define PIN_SERVO_3 0  // Codo (D3)
#define PIN_SERVO_4 14 // Muñeca Pitch (D5) <-- ANTES ERA 2
#define PIN_SERVO_5 2  // Muñeca Roll (D4)  <-- ANTES ERA 14
#define PIN_SERVO_6 4  // Pinza (D2)

// 3. Puerto en el que el ESP8266 escuchará los datos
unsigned int puerto_udp = 4210;

// =================================================================

WiFiUDP udp;
Servo servo1, servo2, servo3, servo4, servo5, servo6;
char buffer_paquete[255]; 

void setup() {
  Serial.begin(115200);
  Serial.println("\nIniciando Brazo Robótico (Modo Wi-Fi)...");

  // --- Conexión a la red Wi-Fi ---
  WiFi.begin(ssid, password);
  Serial.print("Conectando a WiFi...");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n¡Conexión exitosa!");
  Serial.print("Dirección IP del brazo: ");
  Serial.println(WiFi.localIP());

  // --- Inicialización de Servos ---
  servo1.attach(PIN_SERVO_1);
  servo2.attach(PIN_SERVO_2);
  servo3.attach(PIN_SERVO_3);
  servo4.attach(PIN_SERVO_4);
  servo5.attach(PIN_SERVO_5);
  servo6.attach(PIN_SERVO_6);
  
  // --- Iniciar la escucha UDP ---
  udp.begin(puerto_udp);
  Serial.printf("Escuchando datos en el puerto UDP %d\n", puerto_udp);
}

void loop() {
  // 1. Revisa si ha llegado un paquete de datos UDP
  int tamano_paquete = udp.parsePacket();

  if (tamano_paquete) {
    // 2. Lee el paquete y guárdalo en el buffer
    int len = udp.read(buffer_paquete, 255);
    if (len > 0) {
      buffer_paquete[len] = '\0'; 
    }

    // 3. Decodifica el string para extraer los 6 ángulos
    int ang1, ang2, ang3, ang4, ang5, ang6;
    int items_leidos = sscanf(buffer_paquete, "<%d,%d,%d,%d,%d,%d>", 
                              &ang1, &ang2, &ang3, 
                              &ang4, &ang5, &ang6);

    // 4. Si el formato es correcto (6 valores leídos), mueve los servos
    if (items_leidos == 6) {
      servo1.write(ang1); // Base
      servo2.write(ang2); // Hombro
      servo3.write(ang3); // Codo
      servo4.write(ang4); // Muñeca Pitch
      servo5.write(ang5); // Muñeca Roll
      servo6.write(ang6 == 1 ? 45 : 0); // Pinza
    }
  }
}
