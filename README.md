# SOS-ECG-system

This project consists of two main parts: an Arduino sketch for reading ECG data and a Python script for processing the data and sending an SOS message if the ECG values are outside a safe range.

## Arduino Code

### Overview
The Arduino code reads ECG data from an AD8232 sensor connected to GPIO36 (SVP) on an ESP32 or similar microcontroller. The data is then sent to the Serial Monitor.

### Key Components
- `ECG_Pin`: The pin connected to the AD8232 sensor's output.
- `Serial.begin(115200)`: Initializes serial communication at 115200 baud rate.
- `analogRead(ECG_Pin)`: Reads the analog value from the ECG sensor.
- `Serial.println(ECG_value)`: Sends the ECG value to the Serial Monitor.
- `delay(10)`: Introduces a 10ms delay to achieve a 100Hz sampling rate.

### Usage
1. Connect the AD8232 sensor to the ESP32.
2. Upload the code to the ESP32.
3. Open the Serial Monitor to view the ECG values.

## Python Code

### Overview
The Python script reads ECG data from a serial connection, logs it to a file, and checks if the values are within a safe range. If not, it sends an SOS message via WhatsApp and a text-to-speech alert.

### Key Components
- `serial.Serial('COM5', 115200, timeout=1)`: Initializes a serial connection to the ESP32.
- `sos(ecg_value, timestamp)`: Function to check ECG values and send an SOS message if necessary.
- `pywhatkit.sendwhatmsg`: Sends a WhatsApp message with the ECG value and timestamp.
- `pyttsx3`: Converts text to speech to alert about the patient's condition.
- `check_data()`: Reads logged ECG data and checks for anomalies.

### Usage
1. Ensure the ESP32 is connected and the Arduino code is running.
2. Run the Python script.
3. The script will log ECG data to `ecg_data.txt` and send an SOS message if the ECG values are outside the safe range.

## Dependencies

The following Python libraries are required:
- `pyserial`: For serial communication.
- `pywhatkit`: For sending WhatsApp messages.
- `pyttsx3`: For text-to-speech functionality.

Install dependencies using `pip`:

```bash
pip install pyserial pywhatkit pyttsx3
