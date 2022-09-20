import serial

ser = serial.Serial('COM_')
value = ser.readline()

print(value)