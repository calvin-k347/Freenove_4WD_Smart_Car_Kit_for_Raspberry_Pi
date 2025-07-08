from car import Car
from camera import Camera
#importing time and also importing classes from the other file used for first week
import time
#initalizing our_car to be a new instance of a car 
our_car = Car()
camera = Camera()
#assigns all hardware to an object 
our_car.start()


#function for turning the car,moving froward, and reversing the car
def turn_right(t):
    our_car.motor.set_motor_model(3000,3000,-2000,-2000) 
    time.sleep(t)
    our_car.motor.set_motor_model(0,0,0,0)
def turn_left(t):
    our_car.motor.set_motor_model(-2000,-2000,3000,3000) 
    time.sleep(t)
    our_car.motor.set_motor_model(0,0,0,0)
def forward():
    our_car.motor.set_motor_model(800,800,800,800)
def reverse(t):
    our_car.motor.set_motor_model(-800,-800,-800,-800)
    time.sleep(t)
    our_car.motor.set_motor_model(0,0,0,0)
    
# our function for line detection
def boundary():
    #gives a output for infrared every .2 secs
    if (time.time() - our_car.car_record_time) > .2:
        our_car.car_record_time = time.time()
        infrared = our_car.infrared.read_all_infrared()
        #infrared equals 0 we go forward
        print(infrared)
        if infrared == 0:
            forward()
        # reverse towards left to adjust for the track
        elif infrared in [1,3]:
            reverse(.5)
            turn_left(.3)
            dir = "right"
        # reverse towards right to adjust for the track
        elif infrared in [4,6]:
            reverse(.5)
            turn_right(.3)
            dir = "left"
        else:
            reverse(.75)
            


        
try:
    while True:
        boundary()
except KeyboardInterrupt:
    our_car.close()
    print("\nEnd of program")

