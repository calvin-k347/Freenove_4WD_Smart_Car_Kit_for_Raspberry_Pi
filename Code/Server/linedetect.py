from car import Car
#importing time and also importing classes from the other file used for first week
import time
#initalizing our_car to be a new instance of a car 
our_car = Car()
#assigns all hardware to an object 
our_car.start()
# our function for line detection
def boundary():
    #gives a output for infrared every .2 secs
    if (time.time() - our_car.car_record_time) > .2:
        our_car.car_record_time = time.time()
        infrared = our_car.infrared.read_all_infrared()
        #infrared equals 0 we go forward
        if infrared == 0:
            our_car.motor.set_motor_model(600,600,600,600)
            #if not we go back
        else:
            our_car.motor.set_motor_model(-1000,-1000,-1000,-1000)
        
try:
    while True:
        boundary()
except KeyboardInterrupt:
    our_car.close()
    print("\nEnd of program")

