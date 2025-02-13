import serial
import time
import pywhatkit as kit
import pyttsx3
import sys

def sos(ecg_value, timestamp):
    if 20 < int(ecg_value) < 500:
        print(f"ok at {timestamp}")
    else:
        print(f"not ok")
        current_time = time.time()  
        new_time = current_time + 60

        hour = time.strftime("%H", time.localtime(new_time))  
        minute = time.strftime("%M", time.localtime(new_time))  
        second = time.strftime("%S", time.localtime(new_time))  
        
        print(f"Scheduled time: {hour}:{minute}:{second}")  
        
        number = "+917011579672"
        message = f"patient number 301 in danger ecg - {ecg_value} time -{timestamp}"

        kit.sendwhatmsg(number, message, int(hour), int(minute))
        engine = pyttsx3.init()

        engine.setProperty('rate', 150)  
        engine.setProperty('volume', 1)  

        engine.say("patient number 301 in danger")
        engine.runAndWait()
        sys.exit("Program terminated.")  
        
def check_data():
    with open('ecg_data.txt', 'r') as file:
        for line in file:
            content = line.strip().split(":")
            for k in content:
                print(k)
            timestamp = content[0].strip() 
            ecg_value = content[2].strip()
            sos(ecg_value, timestamp)

try:
    ser = serial.Serial('COM5', 115200, timeout=1)
    with open('ecg_data.txt', 'a') as f:
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{timestamp} : {line}\n")
                print(line)
                check_data()
except serial.SerialException as e:
    print(f"Serial connection error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
