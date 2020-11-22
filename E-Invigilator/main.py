from flask import request
from flask import  Flask,render_template
from joblib import load
import pandas as pd
import numpy as np
import os
import glob

# Load library
import cv2
import sys
import numpy as np
import random
import time
import matplotlib
import matplotlib.pyplot as plt
import re
import tensorflow as tf
import random

from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import pylab
# Loading the CNN model
import keras
from tensorflow.keras.models import load_model
loaded_model = load_model('cheat_detection_improved_version4.hdf5',compile=False)

# load the class labels, stored as txt format.
LABELS = open("yolov3.txt").read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(666)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = "yolov3.weights"
configPath = "yolov3.cfg"

# load our YOLO object detector trained on COCO dataset
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

app=Flask(__name__)

basedir = os.path.abspath(os.path.dirname(__file__))

cheat_list=['not_cheating','passing_notes','peeping']

def delete_files(path):
    fileNames = glob.glob(path + r'/*')
    for fileName in fileNames:
            os.remove(fileName)

def prepare_image (img):
    # convert the color from BGR to RGB then convert to PIL array
    cvt_image =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(cvt_image)

    # resize the array (image) then PIL image
    im_resized = im_pil.resize((299, 299))
    img_array = keras.preprocessing.image.img_to_array(im_resized)
    image_array_expanded = np.expand_dims(img_array, axis = 0)
    return keras.applications.mobilenet.preprocess_input(image_array_expanded)

def predict_class(model, images, show = True):
    predicted_names = []
    for student_cv_img in images:
         #student_cv_prepare_img = prepare_image(student_cv_img) 
        student_img = prepare_image(student_cv_img)
        #student_img = keras.preprocessing.image.img_to_array(student_img)                    
        #student_img = np.expand_dims(student_img, axis=0)    
        #student_img /= 255. 
        prediction = loaded_model.predict(student_img)
        index = np.argmax(prediction)
        cheat_list.sort()
        predicted_name = cheat_list[index]
        predicted_names.append(predicted_name)
        #print(predicted_names[0])
        #pylab.imshow(student_img[0])
        #pylab.show()
    return predicted_names

@app.route('/')
def index():    
    delete_files("/static/figures")
    return render_template("index.html")

@app.route('/result')
def result():
    return render_template("result.html")
	
@app.route('/cheatDetectionbyModel', methods=['POST'])
def cheatDetectionbyModel():
    video = request.files.get('uploaded_video') 
    video_name = video.filename
    path = basedir+"/static/uploaded/"
    file_path = path + video_name
    video.save(file_path) # save the video to the local
    cheat_video = cv2.VideoCapture(file_path)
    frame_array = [] # store the frames from the video
    output_frames = []
    i = 1
    while(cheat_video.isOpened()):
        ret, frame = cheat_video.read()
        if ret == False:
          break
        frame_array.append(frame)
    cheat_video.release()
    cv2.destroyAllWindows() 
    thre_confidence = 0.5
    thre_nms = 0.3
    cnt = 0
    last_image = frame_array[0]
    passes=[] # store the frequency of passing notes
    peeps=[] # store the frequency of peeping others
    for image in frame_array:
        original_image = image # original image frame with no bounding boxes
        processed_image = image
        cheat_type_label = []
        pass_ini = 0 # used to count the frequency of students' passing notes
        peep_ini = 0 # used to count the frequency of students' peeping others
        if cnt % 16 == 0:
            (H, W) = image.shape[:2]
            ln = net.getLayerNames()
            ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            start = time.time()
            layerOutputs = net.forward(ln)
            end = time.time()
        
            boxes = []
            confidences = []
            classIDs = []

            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    if confidence > thre_confidence:
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

            idxs = cv2.dnn.NMSBoxes(boxes, confidences, thre_confidence,  thre_nms)

            if len(idxs) > 0:
                for i in idxs.flatten():
                    if classIDs[i] == 0:
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])
                        if x <= 0:
                            x = 1
                        if y <= 0:
                            y = 1
                        if w <= 0:
                            w = 1
                        if h <= 0:
                            h = 1
                        person = original_image[y:y+h, x:x+w] # crop the image on the basis of the original image
                        cheat_list=['not_cheating','passing_notes','peeping']
                        images = []
                        images.append(person)
                    
                        predicted_names = predict_class(loaded_model, images, True)
                        cheat_type_label.append(predicted_names[0])
                tmp = 0
                for i in idxs.flatten():
                    if classIDs[i] == 0 and tmp < len(cheat_type_label):
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])
                        if x <= 0:
                            x = 1
                        if y <= 0:
                            y = 1
                        if w <= 0:
                            w = 1
                        if h <= 0:
                            h = 1
                        if cheat_type_label[tmp] == "passing_notes" or cheat_type_label[tmp] == "peeping":
                            text = cheat_type_label[tmp]
                            if cheat_type_label[tmp] == "passing_notes":
                                pass_ini = pass_ini + 1
                                cv2.rectangle(image, (x, y), (x + w, y + h), (0,0,255), 2, lineType=cv2.LINE_AA)
                                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, lineType=cv2.LINE_AA)
                            else:
                                peep_ini = peep_ini + 1
                                cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,0), 2, lineType=cv2.LINE_AA)
                                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, lineType=cv2.LINE_AA)
                        tmp = tmp + 1
            output_frames.append(image)
        else:
            last_image = output_frames[len(output_frames) - 1]
            output_frames.append(last_image)
        cnt = cnt + 1
        passes.append(pass_ini)
        peeps.append(peep_ini)
        print("Done: " + str(len(output_frames)) + " / " + str(len(frame_array)) + " frames")
    
    video = request.files.get('uploaded_video') 
    video_name = video.filename
    new_file_path = basedir + "/static/processed/"
    new_file_name = new_file_path + video_name + "_processed.mp4"
    result = cv2.VideoWriter(new_file_name, cv2.VideoWriter_fourcc(*'AVC1'), 24, (1920,1080))
    for i in range(len(output_frames)):
        result.write(output_frames[i])
    result.release()


    total_frames = len(frame_array)
    index = []
    name_list = []
    index = range(total_frames)
    plt.cla()
    plt.xlabel('Frame')
    plt.ylabel('Frequency')
    plt.title('Frequency of Peeping Behaviors')
    y = peeps
    plt.plot(index,y,"g",linewidth=1) 
    peep_plot_path = "/static/figures/peeping.jpg?" + str(random.randint(0, 1000))
    plt.savefig("./static/figures/peeping.jpg")
    plt.close()

    plt.cla()
    plt.xlabel('Frame')
    plt.ylabel('Frequency')
    plt.title('Frequency of Passing Notes Behaviors') 
    y2 = passes 
    plt.plot(index,y2,"r",linewidth=1) 
    passing_notes_plot_path = "/static/figures/passing_notes.jpg?" + str(random.randint(0, 1000))
    plt.savefig("./static/figures/passing_notes.jpg")
    plt.close()
    
    new_file_name = "/static/processed/" + video_name + "_processed.mp4"
    return render_template("result.html", file_path = new_file_name, peep_figure = peep_plot_path, passing_notes_figure = passing_notes_plot_path)
	  
if __name__=="__main__":
    app.run(port=2020,host="127.0.0.1",debug=True)