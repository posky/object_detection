#include <Ultrasonic.h>

Ultrasonic sensor(9, 8, 300000);    // (Trig, Echo, max distance us => 30000us ~~ 5m)

int distance = 0;

void setup() {
    Serial.begin(9600);
}

void loop() {
    distance = sensor.Ranging(CM);
    Serial.print("Distance ");
    Serial.print(distance);
    Serial.println(" cm");
    delay(2000);
}