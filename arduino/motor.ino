#include <Ultrasonic.h>
#include <ControlMotor.h>

Ultrasonic sensor(9, 8, 30000);    // (Trig, Echo, max distance us => 30000us ~~ 5m)
ControlMotor control(2, 3, 7, 4, 5, 6);     // (right motor 1, right motor 2, left motor 1, left motor 2, right PWM, left PWM)

int measurement_speed = 5;
long int distance = 0;
int random_value = 0;

void setup() {
    Serial.begin(9600);
}

void loop() {
    control.Motor(150, 1);
    distance = sensor.Ranging(CM);

    delay(measurement_speed);

    // If there is no obstacle
    Serial.print("No obstacle ");
    Serial.println(distance);
    Serial.print("Random ");
    Serial.println(random_value);

    random_value = random(2);

    while (distance < 30) {
        delay(measurement_speed);
        control.Motor(0, 1);
        distance = sensor.Ranging(CM);
        delay(1000);

        Serial.print("Distance ");
        Serial.println(distance);
        Serial.print("Random ");
        Serial.println(random_value);

        if (random_value == 0) {
            control.Motor(170, 100);
            delay(400);
        } else if (random_value == 1) {
            control.Motor(170, 100);
            delay(400);
        }
    }
}