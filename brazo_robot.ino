/*
=================================================================
 PROYECTO: Receptor de Control para Brazo Robótico (6 Ejes)
 VERSIÓN:  1.2 - Usando la librería Servo estándar para ESP8266
=================================================================
 Descripción:
 Este código se ejecuta en una placa tipo ESP8266 y está 
 diseñado para recibir datos de ángulos a través del puerto
 serie desde un script de Python. Utiliza la librería Servo.h
 estándar que viene con el gestor de placas ESP8266.

 El formato de datos esperado es:
 "<base,hombro,codo,pitch,roll,pinza>"
 Ejemplo: "<90,120,45,80,110,1>"

 Cada valor corresponde a un servo, y el último valor (pinza)
 es 0 para abrir y 1 para cerrar.
-----------------------------------------------------------------
*/

// Librería estándar para controlar servos en ESP8266
#include <Servo.h>

// --- OBJETOS SERVO ---
// Se crea un objeto Servo para cada articulación del brazo.
Servo servoBase;
Servo servoHombro;
Servo servoCodo;
Servo servoMunecaPitch;
Servo servoMunecaRoll;
Servo servoPinza;

// --- PINES DE CONEXIÓN (GPIO) ---
// Asegúrate de que estos pines coincidan con tu conexión física.
const int PIN_BASE = 5;          // D1
const int PIN_HOMBRO = 4;        // D2
const int PIN_CODO = 0;          // D3
const int PIN_MUNECA_PITCH = 14; // D5
const int PIN_MUNECA_ROLL = 12;  // D6
const int PIN_PINZA = 2;         // D4

// --- CONFIGURACIÓN DE LA PINZA ---
// Ajusta estos valores según el rango de tu pinza mecánica.
const int ANGULO_PINZA_ABIERTA = 90;  // Ángulo para la pinza abierta
const int ANGULO_PINZA_CERRADA = 0;   // Ángulo para la pinza cerrada

// --- VARIABLES GLOBALES ---
String incomingData; // String para almacenar los datos recibidos del puerto serie
bool dataReady = false; // Bandera para indicar si se ha recibido un paquete completo

void setup() {
  // Iniciar la comunicación serie a la misma velocidad que Python
  Serial.begin(9600);
  Serial.println("--- Controlador de Brazo Robótico Activado ---");
  Serial.println("Esperando datos desde Python...");

  // Conectar cada objeto Servo a su pin correspondiente
  servoBase.attach(PIN_BASE);
  servoHombro.attach(PIN_HOMBRO);
  servoCodo.attach(PIN_CODO);
  servoMunecaPitch.attach(PIN_MUNECA_PITCH);
  servoMunecaRoll.attach(PIN_MUNECA_ROLL);
  servoPinza.attach(PIN_PINZA);

  // Posición inicial de los servos (opcional, pero recomendado)
  servoBase.write(90);
  servoHombro.write(90);
  servoCodo.write(90);
  servoMunecaPitch.write(90);
  servoMunecaRoll.write(90);
  servoPinza.write(ANGULO_PINZA_ABIERTA); // Empezar con la pinza abierta
}

void loop() {
  // 1. Leer los datos del puerto serie
  readSerialData();

  // 2. Si se ha recibido un paquete de datos completo, procesarlo
  if (dataReady) {
    parseAndMoveServos();
  }
}

/**
 * @brief Lee los datos del puerto serie.
 * Detecta los caracteres de inicio ('<') y fin ('>') para ensamblar un paquete de datos completo.
 */
void readSerialData() {
  while (Serial.available() > 0 && !dataReady) {
    char receivedChar = Serial.read();

    // Si el carácter es '>', hemos terminado de recibir el paquete
    if (receivedChar == '>') {
      dataReady = true;
    } 
    // Si no, se añade el carácter al string
    else {
      incomingData += receivedChar;
    }
  }
}

/**
 * @brief Procesa la cadena de datos, extrae los ángulos y mueve los servos.
 */
void parseAndMoveServos() {
  // Eliminar el carácter de inicio '<'
  if (incomingData.startsWith("<")) {
    incomingData = incomingData.substring(1);
  }

  // Imprimir los datos recibidos para depuración
  Serial.print("Datos recibidos: ");
  Serial.println(incomingData);

  // Variables para almacenar los ángulos
  int angulos[6];
  int lastIndex = 0;

  // Extraer cada valor separado por comas
  for (int i = 0; i < 6; i++) {
    int commaIndex = incomingData.indexOf(',', lastIndex);
    if (commaIndex == -1 && i < 5) { // Si faltan comas, el dato es inválido
        Serial.println("Error: Formato de datos incorrecto.");
        resetData();
        return;
    }
    
    String part = incomingData.substring(lastIndex, commaIndex == -1 ? incomingData.length() : commaIndex);
    angulos[i] = part.toInt();
    lastIndex = commaIndex + 1;
  }

  // Mover los servos a los ángulos recibidos
  // El orden de los ángulos es: base, hombro, codo, pitch, roll, pinza
  servoBase.write(angulos[0]);
  servoHombro.write(angulos[1]);
  servoCodo.write(angulos[2]);
  servoMunecaPitch.write(angulos[3]);
  servoMunecaRoll.write(angulos[4]);

  // El último valor (0 o 1) controla la pinza
  if (angulos[5] == 1) {
    servoPinza.write(ANGULO_PINZA_CERRADA);
  } else {
    servoPinza.write(ANGULO_PINZA_ABIERTA);
  }

  // Limpiar variables para la siguiente lectura
  resetData();
}

/**
 * @brief Limpia la variable de datos y la bandera para prepararse para el siguiente paquete.
 */
void resetData() {
    incomingData = "";
    dataReady = false;
}
