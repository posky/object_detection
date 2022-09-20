#include <ControlMotor.h>

ControlMotor control(2, 3, 7, 4, 5, 6);     // right motor1, right motor2, left motor1, left motor2, rightPWM, leftPWM

int speed = 150;

void setup() {
}

void loop() {
    // one motor clockwise, the other motor anti-clockwise
    // forward
    while (speed < 254) {
        speed++;
        control.Motor(speed, 1);
        delay(200);
    }

    // one motor anti-clockwise, the other motor clockwise
    // backward
    control.Motor(-180, 1);
    delay(3000);

    // two motors clockwise
    // turn left
    control.Motor(200, 100);
    delay(3000);

    // two motors anti-clockwise
    // turn right
    control.Motor(200, -100);
    delay(3000);

    // two motors stop
    control.Motor(0, 1);
    delay(3000);

    speed = 150;
}