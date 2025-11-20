import serial.tools.list_ports

print("Puertos disponibles:")
puertos = serial.tools.list_ports.comports()
for puerto in puertos:
    print(f"  - {puerto.device}: {puerto.description}")