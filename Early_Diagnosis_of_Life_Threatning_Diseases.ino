#include <Wire.h>
#include <LiquidCrystal_I2C.h>

#define startMan true
#define alreadyConnected true

LiquidCrystal_I2C lcd(0x27, 16, 2);
int ind = 13;

void start() {
  Serial.begin(115200);
  Serial.println("AT");
  int timeout = 0;
  while ( Serial.available()==0) {
  if( ++timeout > 10000){ // set this to your timeout value in milliseconds
     // your error handling code here
     break;
   }
  }
   Serial.flush();
  
  delay(100);
  Serial.println("AT+CWMODE=1");
  timeout = 0;
  while ( Serial.available()==0) {
  if( ++timeout > 10000){ // set this to your timeout value in milliseconds
     // your error handling code here
     break;
   }
  }
   Serial.flush();
  delay(100);
  Serial.println("AT+CWLAP");
  timeout = 0;
  while ( Serial.available()==0) {
  if( ++timeout > 10000){ // set this to your timeout value in milliseconds
     // your error handling code here
     break;
   }
  }
   Serial.flush();
  delay(2000);
  Serial.println("AT+CWJAP=\"Menlo School\",\"autumnknights\"");
  timeout = 0;
  while ( Serial.available()==0) {
  if( ++timeout > 10000){ // set this to your timeout value in milliseconds
     // your error handling code here
     break;
   }
  }
   Serial.flush();
  delay(10000);
  Serial.println("AT+CIFSR");
  timeout = 0;
  while ( Serial.available()==0) {
  if( ++timeout > 10000){ // set this to your timeout value in milliseconds
     // your error handling code here
     break;
   }
  }
   if(Serial.readString().indexOf("OK" < 0)) {
     start();  
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
  lcd.clear();
  lcd.setCursor(0,0);
  lcd.print("Collect 1");
  String data1 = "";
  while(Serial1.available()==0){}
  data1 = Serial1.readString();
  if(data1.indexOf("sensing") > -1) {
    while(Serial.available() < 1) {}
    data1 = Serial1.readString();
  }
  return data1;
}

String collect2() {
  lcd.clear();
  lcd.setCursor(0,0);
  lcd.print("Collect 2");
  String data2 = "";
  while(Serial2.available()==0){}
  data2 = Serial2.readString();
  if(data2.indexOf("sensing") > -1) {
    while(Serial.available() < 1) {}
    data2 = Serial2.readString();
  }
  return data2;
}

String collect3() {
  if (startMan) {
    return "";
  }
  lcd.clear();
  lcd.setCursor(0,0);
  lcd.print("Collect 3");
  String data3 = "";
  while(Serial3.available()==0){}
  data3 = Serial3.readString();
  if(data3.indexOf("sensing") > -1) {
       while(Serial.available() < 1) {}
       data3 = Serial3.readString();
  }
  return data3;
}

bool publish(String da1, String da2, String da3) {
  String GET = "GET pushingbox?devid=vC6CB20C11C11B9F&S1=" + da1 + "&S2=" + da2 + "&S3=" + da3 + " HTTP/1.1";
  String CIP = "AT+CIPSEND=" + String(GET.length() + 4 + 28 + 23 + 21);
  if (startMan) {
    Serial3.println("AT+CIPMODE=0");
    Serial.println(Serial3.readString());
    delay(100);
    Serial3.println("AT+CIPSTART=\"TCP\",\"api.pushingbox.com\",80");
    Serial.println(Serial3.readString());
    delay(2000);
    Serial3.println(CIP);
    Serial.println(Serial3.readString());
    delay(100);
    Serial3.println(GET);
    Serial3.println("Host: api.pushingbox.com");
    Serial3.println("User-Agent: Arduino");
    Serial3.println("Connection: close");
    Serial.println(Serial3.readString());
    delay(100);
    Serial3.println("AT+CIPCLOSE");
    Serial.println(Serial3.readString());
    delay(1000);
  } else {
    Serial.println("AT+CIPSTART=\"TCP\",\"api.pushingbox.com\",80");
    delay(2000);
    Serial.println("AT+CIPMODE=0");
    delay(1000);
    Serial.println(CIP);
    delay(100);
    Serial.println(GET);
    Serial.println("Host: api.pushingbox.com");
    Serial.println("User-Agent: Arduino");
    Serial.println("Connection: close");
    delay(100);
    Serial.println("AT+CIPCLOSE");
    delay(1000);
  }
  
  return true;
}

void waitFunc(unsigned int next) {
  delay(next - millis());
}

void setup() {
  delay(1000);
  lcd.begin();
  lcd.backlight();
  Serial1.begin(9600);
  Serial2.begin(9600);
  Serial3.begin(9600);
  lcd.setCursor(0,0);
  lcd.print("Start");
  pinMode(10, INPUT);
  digitalWrite(10, HIGH);
  // put your setup code here, to run once:
  if (startMan) {
    Serial3.begin(115200);
    Serial.begin(9600);
    while(digitalRead(10) == HIGH) {
      if (alreadyConnected) {
        delay(10000);
        break;
      }
      if ( Serial3.available() ) {  Serial.write( Serial3.read() );  }

      if ( Serial.available() )  {  Serial3.write( Serial.read() );  }
    }
  } else {
    start();
  }
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
  lcd.setCursor(0,0);
  lcd.print("Start");
  Serial1.println("start");
  d1 = collect1();
  Serial2.println("start");
  d2 = collect2();
  d3 = "";
  if (!startMan) {
    Serial3.println("start");
    d3 = collect3();
  }
  do {
    success = publish(d1, d2, d3);
  } while(!success);
  lcd.clear();
  waitFunc(next);
}


