
const int ECG_Pin = 36;  // Connect OUTPUT pin from AD8232 to SVP (GPIO36)

void setup() {
  Serial.begin(115200);  // Start serial communication
}

void loop() {
  // Read ECG sensor (analog input)
  int ECG_value = analogRead(ECG_Pin);  // Read the ECG signal

  // Print ECG value to Serial Monitor
  Serial.println(ECG_value);

  // Small delay for stability (adjust based on your sampling rate)
  delay(10);  // 10ms delay = 100Hz sampling rate
}