
int ind = 13;
void start() {
  Serial.begin(115200);
  Serial.println("AT+RST");
  delay(100);
  Serial.println("AT");
  delay(100);
  Serial.println("AT+CWMODE=1");
  delay(100);
  Serial.println("AT+CWLAP");
  delay(2000);
  Serial.println("AT+CWJAP=\"Menlo School\",\"autumnknights\"");
  delay(2000);
  Serial.println("AT+CIFSR");
  if(Serial.available()) {
    if(Serial.readString().indexOf("OK" < 0)) {
      start();  
    }
  }
}

void checkConnection(){
  Serial.println("AT+CIFSR");
  if(Serial.available()) {
    if(Serial.readString().indexOf("OK" < 0)) {
      start();  
    }
  }
}

String collect1() {
  unsigned int timeout = 0;
  bool fail1 = false;
  String data1 = "";
  while(Serial1.available()==0){
    if(++timeout > 10000){
      fail1 = true;
      break;
    }
  }
  data1 = Serial1.readString();
  if(data1.indexOf("Collecting" < 0)) {
      data1 = collect1();
  }
  if(fail1) {
    return "";
  }
  return data1;
}

String collect2() {
  unsigned int timeout = 0;
  bool fail2 = false;
  String data2 = "";
  while(Serial2.available()==0){
    if(++timeout > 10000){
      fail2 = true;
      break;
    }
  }
  data2 = Serial2.readString();
  if(data2.indexOf("Collecting" < 0)) {
      data2 = collect2();
  }
  if (fail2) {
    return "";
  }
  return data2;
}

String collect3() {
  unsigned int timeout = 0;
  bool fail3 = false;
  String data3 = "";
  while(Serial3.available()==0){
    if(++timeout > 10000){
      fail3 = true;
      break;
    }
  }
  data3 = Serial3.readString();
  if(data3.indexOf("Collecting" < 0)) {
       data3 = collect3();
  }
  if(fail3) {
    return "";
  }
  return data3;
}

bool publish(String da1, String da2, String da3) {
  return true;
}

void waitFunc(unsigned int next) {
  delay(next - millis());
}

void setup() {
  Serial1.begin(9600);
  Serial2.begin(9600);
  Serial3.begin(9600);
  // put your setup code here, to run once:
  start();
}

String d1 = "";
String d2 = "";
String d3 = ""; 
bool success = true;
int next = 0;
unsigned int interval = 100000;
  
void loop() {
  next = millis() + interval;
  // put your main code here, to run repeatedly:
  Serial1.println("start");
  d1 = collect1();
  Serial2.println("start");
  d2 = collect2();
  Serial3.println("start");
  d3 = collect3();
  do {
    success = publish(d1, d2, d3);
  } while(!success);
  waitFunc(next);
}


