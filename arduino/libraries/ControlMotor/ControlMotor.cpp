#if ARDUINO >= 100
  #include "Arduino.h"
#else
  #include "WProgram.h"
#endif

#include "ControlMotor.h"


#define MOT_LEFT_FORWARD digitalWrite(MotLeft1,HIGH);digitalWrite(MotLeft2, LOW)
#define MOT_LEFT_REVERSE  digitalWrite(MotLeft1, LOW);digitalWrite(MotLeft2, HIGH)

#define MOT_RIGHT_FORWARD digitalWrite(MotRight1,HIGH);digitalWrite(MotRight2, LOW)
#define MOT_RIGHT_REVERSE  digitalWrite(MotRight1, LOW);digitalWrite(MotRight2, HIGH)


ControlMotor::ControlMotor(int MR1,int MR2,int ML1,int ML2,int PWMR,int PWML)
{
   pinMode(MR1,OUTPUT);  // We configure as output the Enable 1 of the Right
   pinMode(MR2,OUTPUT);  // We configure as output the Enable 2 of the Right
   pinMode(ML1,OUTPUT);  // We configure as output the Enable 1 of the Left
   pinMode(ML2,OUTPUT);  // We configure as output the Enable 2 of the Left
   pinMode(PWMR,OUTPUT); // PWM used for Right motor
   pinMode(PWML,OUTPUT); // PWM used for the Left motor

   MotRight1=MR1;  // We store the pin selected for the Right motor Enable 1 in the corresponding variable for later use
   MotRight2=MR2;  // We store the selected pin for the Right motor Enable 2 in the corresponding variable for later use
   MotLeft1=ML1;  // We store the selected pin for the Left motor Enable 1 in the corresponding variable for later use
   MotLeft2=ML2;  // We store the selected pin for the Left motor Enable 2 in the corresponding variable for later use
   pwmR=PWMR;    // We store the selected pin for the Right PWM in the corresponding variable for later use
   pwmL=PWML;    // We store the selected pin for the Left PWM in the corresponding variable for later use
}

// This function is responsible for calculating the speed of the motors according to the angle that we select and the speed.
void ControlMotor::CalculateSpeed(int speed, int turn, int *vel_left, int *vel_right ){

    if( speed < 0 ){speed *= -1;}  // We pass the speed value to positive to make the calculations

    if((turn>=100)||(turn<=-100)){ // At 100% or -100% the robot only turns, so both wheels go at 100% speed
      *vel_left = speed;
      *vel_right = speed;
    }
    else{ // If it is not at 100% turn, we perform the calculations
      // We perform the calculations
      if(turn >= 0){       // The robot moves backwards or forwards straight or to the right according to the turn variable
        turn = 100 - turn; // We invert the values to make the rule of 3 since 0� is 100% of the motors and 100� is 0% of the motors
        *vel_left = speed; // As the rotation is to the right if the variable is greater than 0, the left motor goes at the maximum speed that we set.
        *vel_right = (turn*speed)/100; // We make a rule of three and calculate, if 255(maximum speed) is 100%(of turn), "turn variable" will be X
      }
      else{                // The robot moves straight ahead or to the left depending on the turn variable
        turn += 100;       // We invert the values and pass it to positive to make the rule of 3 since 0� is 100% of the motors and -100� is 0% of the motors
        *vel_right = speed;// As the rotation is to the left if the variable is less than 0, the right motor goes to the maximum speed that we set.
        *vel_left = (turn*speed)/100; // We make a rule of three, passing the turn value to positive and we calculate, if 255(maximum speed) is 100%(of the turn), "turn variable" will be X
      }
    }

}

// This function loads the corresponding values into each motor
void ControlMotor::Motor(int speed, int turn){
  int  vel_left,vel_right;  // We store the variables once processed to load them later to the engines.

  //------------------------------- We make sure that the rotation variable does not exceed 100% or -100% --------------------------------------------//
  if(turn > 100){turn = 100;}
  if(turn < -100){turn = -100;}
  //---------------------------- We make sure that the speed variable does not exceed 255 or -255 --------------------------------------------//
  if(speed > 255){speed = 255;}
  if(speed < -255){speed = -255;}

  CalculateSpeed(speed,turn,&vel_left,&vel_right);  // We perform the calculations and store the values in the variables that we send by pointer
  analogWrite(pwmL,vel_left);  // We load the value of the speed in the left motor
  analogWrite(pwmR,vel_right);  // We load the value of the speed in the right motor

  //-------------------------------- We perform the calculations for the motors according to the received variables ------------------------------------------//
  if (speed >= 0){   // In this case it is understood that the robot must advance
    if(turn >= 100){        // We turn to the right completely at 100% so the right motor turns in the opposite direction (The robot does not advance, it only turns)
      MOT_LEFT_FORWARD;
      MOT_RIGHT_REVERSE;
    }
    else if( turn <= -100){ // We turn to the left completely at 100% so the left motor turns in the opposite direction (The robot does not advance, it only turns)
      MOT_LEFT_REVERSE;
      MOT_RIGHT_FORWARD;
    }
    else{                   // The robot advances or rotates while continuing forward, so both motors rotate in a positive direction.
      MOT_LEFT_FORWARD;
      MOT_RIGHT_FORWARD;
    }
  }
  else{                  // In this case it is understood that the robot must go back
    if(turn >= 100){        // We turn to the right (Seen from behind) completely at 100% so the right motor turns in the opposite direction (The robot does not go back, it only turns)
      MOT_LEFT_REVERSE;
      MOT_RIGHT_FORWARD;
    }
    else if( turn <= -100){ // We turn to the left (Seen from behind) completely at 100% so the left motor turns in the opposite direction (The robot does not go back, it only turns)
      MOT_LEFT_FORWARD;
      MOT_RIGHT_REVERSE;
    }
    else{                   // The robot goes backwards or turns continuously backwards so both motors turn in the negative direction.
      MOT_LEFT_REVERSE;
      MOT_RIGHT_REVERSE;
    }
  }
}
