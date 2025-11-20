import serial
import time

try:
    ser = serial.Serial('COM5', 115200, timeout=1)
    print("Conectado a COM5")
    time.sleep(2)
    
    print("Enviando datos de prueba...")
    for i in range(5):
        datos = f"<90,90,90,90,90,90,90,90,90,90>"
        print(f"Enviando: {datos}")
        ser.write(datos.encode('utf-8'))
        time.sleep(1)
    
    ser.close()
    print("Test completado")
    
except serial.SerialException as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"Error inesperado: {e}")