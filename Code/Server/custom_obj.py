import yolov5
import cv2
import numpy as np
import os
from led import Led
from car import Car
from camera import Camera
import threading
import sys
import math
#importing time and also importing classes from the other file used for first week
import time
#initalizing our_car to be a new instance of a car 
our_car = Car()
model_name = 'Yolov5_models'
yolov5_model = 'best.pt'
model_labels = 'balls5n.txt'

CWD_PATH = os.getcwd()
PATH_TO_LABELS = os.path.join(CWD_PATH,model_name,model_labels)
PATH_TO_YOLOV5_GRAPH = os.path.join(CWD_PATH,model_name,yolov5_model)
model = yolov5.load(PATH_TO_YOLOV5_GRAPH)
#initalizing an instance of our camera
camera = Camera()
#assigns all hardware to an object 
our_car.start()
led = Led()
hue_list = []
found_ball = False
last_turn = None
#function for turning the car,moving froward, and reversing the car
def turn_right(t):
    our_car.motor.set_motor_model(3000,3000,-2000,-2000) 
    time.sleep(t)
    our_car.motor.set_motor_model(0,0,0,0)
def turn_left(t):
    our_car.motor.set_motor_model(-2000,-2000,3000,3000) 
    time.sleep(t)
    our_car.motor.set_motor_model(0,0,0,0)
def forward(obj,t, force=None):
    camera.save_image("test.jpg")
    image = cv2.imread("test.jpg") 
    obj_info = find_ball(image)
    if obj_info and obj_info["color"] == obj:
        print("BALL")
        infrared = our_car.infrared.read_all_infrared()
        if obj != "ball":
            return
        while infrared == 0:
            time.sleep(.1)
            camera.save_image("test.jpg")
            image = cv2.imread("test.jpg") 
            obj_info = find_ball(image)
            if obj_info["dir_from_center"] > 0:
                turn_left(.05)
            elif obj_info["dir_from_center"] < 0:
                turn_right(.05)
            infrared = our_car.infrared.read_all_infrared()
            our_car.motor.set_motor_model(1200,1200,1200,1200)
        our_car.motor.set_motor_model(0,0,0,0)
        print(f"I FOUND {obj}")
        return True
    obj_in_the_way = obj_info and ((obj_info["type"] == "balls" and obj_info["color"] != obj and obj_info["dir_from_center"] == 0) or (obj_info["type"] != obj))
    box_detected = our_car.sonic.get_distance() < 10
    print(obj_info, "| way: ", obj_in_the_way, "| box: ", box_detected)
    if obj_in_the_way:
        turn_left(.05)
    elif box_detected:
        reverse(.5)
        turn_left(.5)
    our_car.motor.set_motor_model(1200,1200,1200,1200)

    time.sleep(t)
def reverse(t):
    our_car.motor.set_motor_model(-800,-800,-800,-800)
    time.sleep(t)
    our_car.motor.set_motor_model(0,0,0,0)
# function to find balls
def find_ball(img):
    frame = img.copy()
    hue_value = None
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    model_name = 'Yolov5_models'
    yolov5_model = 'best.pt'
    model_labels = 'balls5n.txt'

    CWD_PATH = os.getcwd()
    PATH_TO_LABELS = os.path.join(CWD_PATH,model_name,model_labels)
    PATH_TO_YOLOV5_GRAPH = os.path.join(CWD_PATH,model_name,yolov5_model)

    # Import Labels File
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Initialize Yolov5
    

    stride, names, pt = model.stride, model.names, model.pt
    print('stride = ',stride, 'names = ', names)
    #model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup

    min_conf_threshold = 0.25
    # set model parameters
    model.conf = 0.25  # NMS confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    model.agnostic = False  # NMS class-agnostic
    model.multi_label = True # NMS multiple labels per box
    model.max_det = 1000  # maximum number of detections per image

    results = model(frame)
    predictions = results.pred[0]

    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    classes = predictions[:, 5]
    # Draws Bounding Box onto image
    results.render() 

    # Initialize frame rate calculation
    frame_rate_calc = 30
    freq = cv2.getTickFrequency()

    imW, imH = int(400), int(300)
    #imW, imH = int(640), int(640)
    #frame_resized = cv2.resize(frame_rgb, (imW, imH))
    #input_data = np.expand_dims(frame_resized, axis=0)

    max_score = 0
    max_index = 0
    
    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        curr_score = scores.numpy()
        # Found desired object with decent confidence
        if ((curr_score[i] > min_conf_threshold) and (curr_score[i] <= 1.0)):
            print('Class: ',labels[int(classes[i])],' Conf: ', curr_score[i])
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            xmin = int(max(1,(boxes[i][0])))
            ymin = int(max(1,(boxes[i][1])))
            xmax = int(min(imW,(boxes[i][2])))
            ymax = int(min(imH,(boxes[i][3])))
            height = ymin - ymax
            # frame appears to use y-x coordinates when compared to a JPEG viewer
            croppedImage = frame[ymin:ymax,xmin:xmax]
            center_x = imW // 2
            center_y = imH // 2
            
            
            hsv_pixel = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2HSV)
            hue_channel = hsv_pixel[int(len(hsv_pixel)*(3/8)):int(len(hsv_pixel)*(5/8)),int(len(hsv_pixel[0])*(3/8)):int(len(hsv_pixel[0])*(5/8)), 0]
            hue_value = int(hue_channel.mean())

            #print('Hue: ',hue_value)
            img_center_x = xmin + (xmax - xmin) // 2
            img_center_y = ymin + (ymax - ymin) // 2
            hypotenuse = int(math.sqrt(height ** 2 + (center_x - img_center_x) ** 2 ))
            dist = center_x - img_center_x
            # Draw label            
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s %s %s' % (f" {object_name}", f"{hue_for_color(hue_value)}", "r"+str(int(dist/height))) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            cv2.rectangle(img, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(img, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
            #if cType.getType() == "ball":
            
            cv2.circle(img, (img_center_x, img_center_y) , 5, (0,0,255), -1)
            cv2.circle(img, (center_x, center_y) , 5, (0,255,0), -1)
                
            # Record current max
            max_score = curr_score[i]
            max_index = i
            cv2.imwrite('video.jpg', img)
            return {"type": object_name, "color": hue_for_color(hue_value), "dir_from_center": int(dist/height)  }

            

    # Write Image (with bounding box) to file
    cv2.imwrite('video.jpg', img)
    return None
def success(color):
    for i in range(15):
        light_up(color)
        time.sleep(.1)
        led.colorBlink(0)
        time.sleep(.1)
def hue_for_color(hue):
    color = hue
    if 20 < hue <= 60:
        color = "yellow"
    elif 60 < hue <= 90:
        color = "green"
    elif 90 < hue <= 110:
        color = "blue"
    elif 110 < hue <= 180:
        color = "red"
    return color
def light_up(color):
    if color == "blue":
        led.ledIndex(0x01, 0,   0,   255)
    if color == "red":
        led.ledIndex(0x01, 255,   0,   0)
    if color == "green":
        led.ledIndex(0x01, 0,   255,  0)
    if color == "yellow":
        led.ledIndex(0x01, 250,   255,  0)
def relocate():
    pass
def survey(obj_list):
    camera.start_stream()
    try:
        for obj in obj_list:
            while True:
                if (time.time() - our_car.car_record_time) > .2:
                    # get sensor readings
                    our_car.car_record_time = time.time()
                    infrared = our_car.infrared.read_all_infrared()
                    sonic_dist = our_car.sonic.get_distance()
                    time.sleep(.25)
                    if infrared == 0:
                        found = forward(obj, .2)
                        our_car.motor.set_motor_model(0,0,0,0)
                    else:
                        reverse(.5)
                        turn_right(.5)
                    if found:
                        break


                    


    except KeyboardInterrupt:
        our_car.close()
        led.colorBlink(0) 
        led.close()
def test():
    try:

        while True:
            dist = our_car.sonic.get_distance()
            print(dist)
    except KeyboardInterrupt:
        our_car.motor.set_motor_model(0,0,0,0)
if __name__ == "__main__":
    color_seq = []
    args = sys.argv[1:]
    for arg in args:
        color_seq.append(arg)
    
    #line = threading.Thread(target=linedetect)
    #line.start()
    survey(color_seq)
