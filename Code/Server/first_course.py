def test_Motor(car): 
    import time 
    PWM = car  

    def turn_right(t):
        PWM.set_motor_model(3000,3000,-2000,-2000) 
        time.sleep(t)
        PWM.set_motor_model(0,0,0,0)
    def turn_left(t):
        PWM.set_motor_model(-2000,-2000,3000,3000) 
        time.sleep(t)
        PWM.set_motor_model(0,0,0,0)
    def forward(t):
        PWM.set_motor_model(3000,3000,3000,3000)
        time.sleep(t)
        PWM.set_motor_model(0,0,0,0)


    try:
        i = 1
        with open("instruction.txt") as f:
            i = 1
            for line in f:
                instruct = line.strip().split()
                print(f"step {i}: {instruct}")
                i += 1
                dir = instruct[0]
                tx = float(instruct[1])
                if dir == "forward":
                    forward(tx)
                if dir == "right":
                    turn_right(tx)
                if dir == "left":
                    turn_left(tx)
            
        '''first part good
        PWM.set_motor_model(1500,1500,1500,1500)         
        time.sleep(2.8)

        PWM.set_motor_model(2000,2000,-2000,-2000)    
        time.sleep(.5)
        PWM.set_motor_model(1000,1000,1000,1000)
        time.sleep(1.1)
        PWM.set_motor_model(3000,3000,-2000,-2000) 
        time.sleep(.25)
        PWM.set_motor_model(1000,1000,1000,1000)
        time.sleep(1.5)
        PWM.set_motor_model(2000,2000,-2000,-2000)     #Right    
        time.sleep(1.3)
        PWM.set_motor_model(3000,3000,2700,2700)
        time.sleep(2)
        PWM.set_motor_model(-2000,-2000,2000,2000)     #Left 
        time.sleep(1)'''






    except KeyboardInterrupt:
        print ("\nEnd of program")
        PWM.close()
    finally:
        PWM.close() # Close the PWM instance
test_Motor()