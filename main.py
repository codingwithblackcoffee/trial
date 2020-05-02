import cv2
import numpy as np

#loading the model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

#loading classes
classes = []
permanent_boxes = []
with open("coco.names","r") as f:
    classes=[line.strip() for line in f.readlines()]

#input and output layers from yolo-v3
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
font = cv2.FONT_HERSHEY_COMPLEX

#Video Capture
cap = cv2.VideoCapture(0)
frame_count = 1
while (True):
    boxes = []
    confidences= []
    class_ids = []
    ret, img = cap.read()
    demoimg = cv2.imread("demo2.jpeg")
    seat_taken = [[0,0,180],[0,0,180]]

    #storing original dimentions of the image
    height, width, channels = img.shape

    #creating blob from the image
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True, crop=False)

    #passing the blob through the detection model and storing all relevant information
    net.setInput(blob)
    outs = net.forward(output_layers)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)   
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0]*width)
                center_y = int(detection[1] * height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                """cv2.circle(img,(center_x,center_y), 10, (0,255,0), 2)"""
                x = int(center_x -w / 2)
                y = int(center_y - h/2)
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    number_objects = len(boxes)
    for i in range(number_objects):
        color_box = (0,255,0)
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        for box_dim in permanent_boxes:
            x1,y1,w1,h1 = box_dim
            if((abs((x +(w/2)) - (x1 + (w1/2))) < w/2) and (abs((y +(h/2)) - (y1 + (h1/2))) < h/2)):
                if(label == "person"):
                    if(permanent_boxes.index(box_dim) == 1):
                        if(permanent_boxes[0][0] > x1):
                            seat_taken[1] = [0,180,0]
                        else:
                            seat_taken[0] = [0,180,0]
                    if(permanent_boxes.index(box_dim) == 0):
                        if(permanent_boxes[1][0] > x1):
                            seat_taken[1] = [0,180,0]
                        else:
                            seat_taken[0] = [0,180,0]                        
                
        
        if(label != "chair"):
            cv2.rectangle(img, (x,y), (x+w,y+h),color_box, 2)
            cv2.putText(img, label, (x,y+30), font, 1, (0,0,0), 3)
            pass
        elif(frame_count == 1):
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            permanent_boxes.append(boxes[i])
    if frame_count != 1 :
        for box_dim in permanent_boxes:
            x,y,w,h = box_dim
            cv2.rectangle(img, (x,y), (x+w,y+w), (0,255,0), 2)

    x,y,w,h = 440, 365, 335, 615
    x1,y1,w1,h1 = 860, 365, 335, 615
    sub_img = demoimg[y:y+h, x:x +w]
    sub_img2 = demoimg[y1:y1+h1, x1:x1 +w1]
    white_rect = np.ones(sub_img.shape, dtype=np.uint8) 
    white_rect1 = np.ones(sub_img.shape, dtype=np.uint8) 
    for matrix in range(len(white_rect)):
        for matrix2 in range(len(white_rect[matrix])):
            white_rect[matrix][matrix2] = seat_taken[0]
    for matrix in range(len(white_rect1)):
        for matrix2 in range(len(white_rect1[matrix])):
            white_rect1[matrix][matrix2] = seat_taken[1]        
        
    res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
    res2 = cv2.addWeighted(sub_img2, 0.5, white_rect1, 0.5, 1.0)
    demoimg[y:y+h, x:x+w] = res
    demoimg[y1:y1+h1, x1:x1+w1] = res2
    cv2.imshow("image",demoimg)
    frame_count +=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    """cv2.waitKey(0)"""
    """cv2.destroyAllWindows()"""   
