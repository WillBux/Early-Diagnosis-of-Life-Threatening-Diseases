int pin = A0;

String sense() {
  digitalWrite(13, HIGH);
  delay(1000);
  while(digitalRead(10) == HIGH) { };
  String sensor = "Weight%20";
  int raw = analogRead(pin);
  raw += analogRead(pin);
  raw += analogRead(pin);
  raw = raw/3.0;
  digitalWrite(13, LOW);
  return sensor + String(raw);
  
}

void setup() {
  Serial.begin(9600);
  pinMode(13, OUTPUT);
  pinMode(10, INPUT);
  digitalWrite(10, HIGH);
}

void loop() {
  if (Serial.readString().indexOf("start") > -1) {
    Serial.println("sensing");
    Serial.println(sense());
  }
}
