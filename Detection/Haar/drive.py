import numpy as np
import cv2
import time
from time import sleep
import RPi.GPIO as GPIO

MOTOR1_EN = 21
MOTOR1_A = 20
MOTOR1_B = 16

MOTOR2_EN = 26
MOTOR2_A = 19
MOTOR2_B = 13

# Configure les pins
GPIO.setmode(GPIO.BCM)

GPIO.setup(MOTOR1_EN, GPIO.OUT)
GPIO.setup(MOTOR1_A, GPIO.OUT)
GPIO.setup(MOTOR1_B, GPIO.OUT)

GPIO.setup(MOTOR2_EN, GPIO.OUT)
GPIO.setup(MOTOR2_A, GPIO.OUT)
GPIO.setup(MOTOR2_B, GPIO.OUT)

def fwd():
    GPIO.output(MOTOR1_EN, GPIO.HIGH)
    GPIO.output(MOTOR1_A, GPIO.LOW)
    GPIO.output(MOTOR1_B, GPIO.HIGH)

    GPIO.output(MOTOR2_EN, GPIO.HIGH)
    GPIO.output(MOTOR2_A, GPIO.HIGH)
    GPIO.output(MOTOR2_B, GPIO.LOW)

def stop():
    GPIO.output(MOTOR1_EN, GPIO.HIGH)
    GPIO.output(MOTOR1_A, GPIO.LOW)
    GPIO.output(MOTOR1_B, GPIO.LOW)

    GPIO.output(MOTOR2_EN, GPIO.HIGH)
    GPIO.output(MOTOR2_A, GPIO.LOW)
    GPIO.output(MOTOR2_B, GPIO.LOW)


def turn_right():
    GPIO.output(MOTOR1_EN, GPIO.HIGH)
    GPIO.output(MOTOR1_A, GPIO.HIGH)
    GPIO.output(MOTOR1_B, GPIO.LOW)

    GPIO.output(MOTOR2_EN, GPIO.HIGH)
    GPIO.output(MOTOR2_A, GPIO.HIGH)
    GPIO.output(MOTOR2_B, GPIO.LOW)


stop_cascade = cv2.CascadeClassifier('stop.xml')
turn_right_cascade = cv2.CascadeClassifier('turn_right.xml')

cap = cv2.VideoCapture(0)
time.sleep(1)

frame_rate_calc = 1
freq = cv2.getTickFrequency()

while 1:

		fwd()
		sleep(1)
        t1 = cv2.getTickCount()
        ret, img = cap.read()
        img = cv2.flip(img, 0)
        img = cv2.flip(img, 1)
        resize = cv2.resize(img,(320,240))
        gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)

        stop = stop_cascade.detectMultiScale(gray,1.4,6)
        turn_right = turn_right_cascade.detectMultiScale(gray,1.08,4)

        for (x,y,w,h) in stop:
                cv2.rectangle(resize,(x,y),(x+w,y+h),(255,0,0),2)
                print('Found stop')
                stop()
                sleep(1)

        for (x,y,w,h) in turn_right:
                cv2.rectangle(resize,(x,y),(x+w,y+h),(0,255,0),2)
                print('Found turn_right')
                turn_right()
                sleep(2)

        cv2.putText(resize,"FPS: {0:.2f}".format(frame_rate_calc),(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
        cv2.imshow('img',resize) 

        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1
       
        if cv2.waitKey(30) & 0xff == 27:
                break

cap.release()
cv2.destroyAllWindows()