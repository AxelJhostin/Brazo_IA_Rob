#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// Crea el objeto para controlar el HW-170 (PCA9685)
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// Límites de pulso para tus servos (ajusta si los tuyos son diferentes)
#define SERVOMIN  150 // Pulso mínimo (~0 grados)
#define SERVOMAX  600 // Pulso máximo (~180 grados)

// --- CONFIGURACIÓN DE SERVOS ---

// Tu script de Python envía 10 valores
const int NUM_SERVOS = 10;

// Array para guardar los ángulos recibidos
int angulos[NUM_SERVOS];

/*
¡¡¡RECORDATORIO CRUCIAL!!!
Asegúrate de que este array coincida con tu cableado físico.
El orden DEBE coincidir con el que envía Python:
[0] = proximidad (Base)
[1] = hombro
[2] = codo
[3] = pitch (Inclinación)
[4] = roll (Rotación)
[5] = pulgar
[6] = indice
[7] = medio
[8] = anular
[9] = meñique
*/
// Edita los números de abajo (0-15) según los pines de tu PCA9685
int canalesServos[NUM_SERVOS] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };


// --- FIN DE CONFIGURACIÓN ---


/**
 * @brief Convierte grados (0-180) a pulsos PWM (SERVOMIN-SERVOMAX)
 */
int gradosAPulso(int angulo) {
  // Mapea el ángulo de 0-180 a los límites de pulso
  return map(angulo, 0, 180, SERVOMIN, SERVOMAX);
}

void setup() {
  // Inicia la comunicación serial. Debe ser 115200.
  Serial.begin(115200);
  Serial.println("Control de Servos por Serial - Listo.");

  pwm.begin();
  pwm.setPWMFreq(50); // Frecuencia estándar de 50 Hz para servos

  // Centra todos los servos a 90 grados al iniciar
  for (int i = 0; i < NUM_SERVOS; i++) {
    angulos[i] = 90; // Guarda el estado inicial
    pwm.setPWM(canalesServos[i], 0, gradosAPulso(90));
  }
}

/**
 * @brief Mueve todos los servos a las posiciones guardadas en el array 'angulos'
 */
void moverServos() {
  for (int i = 0; i < NUM_SERVOS; i++) {
    int pulso = gradosAPulso(angulos[i]);
    pwm.setPWM(canalesServos[i], 0, pulso);
  }
}

/**
 * @brief Analiza la cadena de datos (ej: "90,80,70...") y actualiza el array 'angulos'
 */
void procesarDatos(String data) {
  int angleIndex = 0;

  while (data.length() > 0 && angleIndex < NUM_SERVOS) {
    int comaIndex = data.indexOf(','); // Busca la siguiente coma

    String valorStr;
    if (comaIndex == -1) {
      // No hay más comas, es el último valor
      valorStr = data;
      data = ""; // Limpiar la cadena para salir del bucle
    } else {
      // Extraer el valor antes de la coma
      valorStr = data.substring(0, comaIndex);
      // Quitar este valor y la coma de la cadena
      data = data.substring(comaIndex + 1);
    }

    // Convertir el valor a entero y guardarlo
    angulos[angleIndex] = valorStr.toInt();
    angleIndex++;
  }
}

/**
 * @brief Bucle principal: Escucha datos seriales
 */
void loop() {
  // 1. Revisa si hay datos disponibles
  if (Serial.available() > 0) {
    
    // 2. Lee el primer caracter, buscando el inicio '<'
    char startChar = Serial.read();
    
    if (startChar == '<') {
      // 3. Si lo encuentra, lee todo hasta el final '>'
      String dataString = Serial.readStringUntil('>');

      // 4. Procesa la cadena de datos
      procesarDatos(dataString);

      // 5. Mueve los servos a las nuevas posiciones
      moverServos();
    }
    // Si el caracter no es '<', se ignora y el bucle sigue.
  }
}