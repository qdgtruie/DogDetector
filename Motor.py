import time


class ServoMotor:

    def __init__(self, **engine_kwargs):
        # define pulse offset of servo
        self.OFFSE_DUTY = 0.5
        # define pulse duty cycle for minimum angle of servo
        self.SERVO_MIN_DUTY = 2.5 + self.OFFSE_DUTY
        # define pulse duty cycle for maximum angle of servo
        self.SERVO_MAX_DUTY = 12.5 + self.OFFSE_DUTY
        self.servoPin = 12
        self.MAX_ANGLE = 225

    def mapAngle(self, value, fromLow, fromHigh, toLow, toHigh):
        return (toHigh - toLow) * (value - fromLow) \
               / (fromHigh - fromLow) + toLow

    def setup(self):
        import RPi.GPIO as GPIO
        global p
        GPIO.setmode(GPIO.BOARD)  # Numbers GPIOs by physical location
        GPIO.setup(self.servoPin, GPIO.OUT)  # Set servoPin's mode is output
        GPIO.output(self.servoPin, GPIO.LOW)  # Set servoPin to low

        p = GPIO.PWM(self.servoPin, 50)  # set Frequece to 50Hz
        p.start(0)  # Duty Cycle = 0

    # make the servo rotate to specific angle (0-180 degrees)
    def Rotate(self, angle):
        if (angle < 0):
            angle = 0
        elif (angle > self.MAX_ANGLE):
            angle = self.MAX_ANGLE
            # map the angle to duty cycle and output it
        value = self.mapAngle(angle, 0, self.MAX_ANGLE,
                              self.SERVO_MIN_DUTY, self.SERVO_MAX_DUTY)
        p.ChangeDutyCycle(value)

    def Standby(self):
        p.ChangeDutyCycle(self.SERVO_MIN_DUTY)

    def loopOnce(self):
        p.start(0)
        # make servo rotate from 0 to 180 deg
        for dc in range(0, self.MAX_ANGLE + 1, 1):
            self.Rotate(dc)  # Write to servo
            time.sleep(0.001)
        print("[DEBUG] Angle reached {0!s} !".format(self.MAX_ANGLE))
        time.sleep(2)

        for dc in range(225, -1, -1):  # make servo rotate from 180 to 0 deg
            self.Rotate(dc)
            time.sleep(0.001)

        print("[DEBUG] Angle reached 0 !")
        p.start(0)
        time.sleep(2)

    def loop(self):
        while True:
            self.loopOnce()

    def destroy(self):
        import RPi.GPIO as GPIO
        p.stop()
        GPIO.cleanup()


if __name__ == '__main__':  # Program start from here
    print('Program is starting...')
    motor = ServoMotor()
    motor.setup()
    try:
        motor.loopOnce()
        # When 'Ctrl+C' is pressed,
        # the child program destroy() will be  executed.
    except KeyboardInterrupt:
        motor.destroy()
    finally:
        motor.destroy()
