import yolov5
import cv2
import numpy as np
import os

from car import Car
from camera import Camera
#importing time and also importing classes from the other file used for first week
import time
#initalizing our_car to be a new instance of a car 
our_car = Car()

#initalizing an instance of our camera
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
# function to find balls
def find_ball(img):
    frame = img.copy()
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    model_name = 'Yolov5_models'
    yolov5_model = 'balls5n.pt'
    model_labels = 'balls5n.txt'

    CWD_PATH = os.getcwd()
    PATH_TO_LABELS = os.path.join(CWD_PATH,model_name,model_labels)
    PATH_TO_YOLOV5_GRAPH = os.path.join(CWD_PATH,model_name,yolov5_model)

    # Import Labels File
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Initialize Yolov5
    model = yolov5.load(PATH_TO_YOLOV5_GRAPH)

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

    #imW, imH = int(400), int(300)
    imW, imH = int(640), int(640)
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
                       
            # frame appears to use y-x coordinates when compared to a JPEG viewer
            croppedImage = frame[ymin:ymax,xmin:xmax]



            hsv_pixel = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2HSV)

            #print('Hue: ',hue_value)
  
            # Draw label            
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%: %s' % (object_name, int(curr_score[i]*100),"hue_value") # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            cv2.rectangle(img, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(img, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
            #if cType.getType() == "ball":
                
            # Record current max
            max_score = curr_score[i]
            max_index = i

    # Write Image (with bounding box) to file
    cv2.imwrite('video.jpg', img)
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
        # reverse towards right to adjust for the track
        elif infrared in [4,6]:
            reverse(.5)
            turn_right(.3)
        else:
            reverse(.75)
            


        
'''try:
    while True:
        boundary()
except KeyboardInterrupt:
    our_car.close()
    print("\nEnd of program")'''

camera.start_stream()
time.sleep(3)
print("Capture image...")
camera.save_image(filename="test.jpg")   
image = cv2.imread("test.jpg")           # Capture and save an image
find_ball(image)
camera.close()
our_car.close()

