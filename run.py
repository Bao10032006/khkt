from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import imutils
import matplotlib.pyplot as plt
from imutils.video import VideoStream
import tensorflow as tf
import argparse
import facenet
import sys
import pickle
import collections
from sklearn.svm import SVC
from PIL import Image, ImageOps
import csv
import cv2
import pandas as pd
import numpy as np
import threading
import time
from tracker import excuteCMD
from tracker import openLink
from ultralytics import YOLO
from tracker1 import Tracker
import os
import re
import telegram
from datetime import datetime
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import datetime
from twilio.rest import Client 
import cvzone
import threading
import cvzone
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=140)

# Open a video capture
points=[]
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_FLAG_LBUTTON :  
        point = [x, y]
        
        points.append(point)
        print(points)
  
        
def sendtele():        
    my_token = "6189869386:AAHIb6rcUzj43T3NQeo0_znrq2RbbMdUuzQ"
    bot = telegram.Bot(token=my_token)
    bot.sendPhoto(chat_id="6231784055", photo=open("alert.png", "rb"), caption="còn người trên xe")
def sendtele2(name):        
    my_token = "6189869386:AAHIb6rcUzj43T3NQeo0_znrq2RbbMdUuzQ"
    bot = telegram.Bot(token=my_token)
    bot.sendPhoto(chat_id="6231784055", photo=open("hocsinh.png", "rb"), caption="học sinh "+name+" đã xuống xe")    
def sendtele3(names):        
    my_token = "6189869386:AAHIb6rcUzj43T3NQeo0_znrq2RbbMdUuzQ"
    bot = telegram.Bot(token=my_token)
    bot.sendPhoto(chat_id="6231784055", photo=open("hocsinh1.png", "rb"), caption="những em "+names[1]+" chưa đi học")
# Hàm lấy thông tin GPS
def gps():
   excuteCMD()
   time.sleep(1)
   openLink()
# cv2.namedWindow('RGB')
# cv2.setMouseCallback('RGB', RGB)F
model=YOLO("E:/2025/khkt/khkt/Models/best.pt")
my_file = open("E:/2025/khkt/khkt/coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
global names, hocsinhdadenlop 
names = []
hocsinhdadenlop=[]
with open('E:/2025/khkt/khkt/ttll.txt', 'r') as file:
    lines = file.readlines()

# Lặp qua từng dòng và phân tách tên và số điện thoại
for line in lines:
    namehocsinh, phone_number = line.strip().split()
    if namehocsinh=="XuânAnh":
        namehocsinh="Xuân Anh"
    names.append(namehocsinh)
    print ("list", names)
tracker=Tracker()
area1=[(536, 126),(518, 382),(647, 391),(640, 135)]
area2=[[580, 121], [564, 381], [481, 380], [504, 116]]
overlap_area = list(set(map(tuple, area1)) & set(map(tuple, area2)))  # Chuyển list thành tuple

er={}
counter1=[]
ex={}
counter2=[]
state = {}
last_alert_time = None
global name

def countpeople():
    video_capture = cv2.VideoCapture(r"E:/2025/khkt/khkt/peoplecount_2.mp4")
    while True:
        with open("E:/2025/khkt/khkt/test_1/output.txt", "r") as file:
            content = file.readlines()

    # Khởi tạo danh sách để chứa các số thập phân
        
        decimal_numbers = []
        
    # Lặp qua từng dòng trong file
        for line in content:
        # Sử dụng regex để tìm tất cả các số thập phân trong dòng
            decimals = re.findall(r'\d+\.\d+', line)
            # Chuyển các chuỗi số thành số thập phân (float) và thêm vào danh sách
            decimal_numbers.extend(float(num) for num in decimals)
        ret, framepeople = video_capture.read()
        if not ret:
            break
        framepeople=cv2.resize(framepeople,(1028,500))
        results=model.predict(framepeople)
        a=results[0].boxes.data
        px=pd.DataFrame(a).astype("float")
    
        list=[]    
        for index,row in px.iterrows():
    #        print(row)
    
            x1=int(row[0])
            y1=int(row[1])
            x2=int(row[2])
            y2=int(row[3])
            
            d=int(row[5])
            c=class_list[d]
            list.append([x1,y1,x2,y2])
            cv2.rectangle(framepeople,(x1,y1),(x2,y2),(0,255,0),2)
        bbox_idx=tracker.update(list)
        for bbox in bbox_idx:
            x1,y1,x2,y2,id=bbox
            cx=int(x1+x2)//2
            cy=int(y1+y2)//2
            centroid=(cx,cy)
            cv2.circle(framepeople, centroid, 5, (255,0,0), -1)
            in_area1 = cv2.pointPolygonTest(np.array(area1, np.int32), (cx, cy), False) >= 0
            in_area2 = cv2.pointPolygonTest(np.array(area2, np.int32), (cx, cy), False) >= 0
            in_overlap = in_area1 and in_area2
    # Lấy trạng thái hiện tại của điểm
            current_area = state.get(id, None)

    # Xử lý chuyển đổi trạng thái
            if current_area is None:  # Nếu chưa có trạng thái
                if in_area1 and not in_overlap:
                    state[id] = 'area1'
                elif in_area2 and not in_overlap:
                    state[id] = 'area2'
                elif in_overlap:
                    state[id] = 'overlap'

            elif current_area == 'area1':  # Nếu đang ở area1
                if in_overlap:  # Đi vào vùng chồng lấp
                    state[id] = 'overlap'
                    cv2.rectangle(framepeople, (x1, y1), (x2 + x1, y2 + y1), (0, 255, 0), 3)
                    cvzone.putTextRect(framepeople, f'{id} Entering', (cx, cy), 2, 2)
                    cv2.circle(framepeople, (cx, cy), 5, (0, 255, 0), -1)
                    if id not in counter1:
                        counter1.append(id)
                        
            elif current_area == 'area2':  # Nếu đang ở area2
                if in_overlap:  # Đi vào vùng chồng lấp
                    state[id] = 'overlap'
                    cv2.rectangle(framepeople, (x1, y1), (x2 + x1, y2 + y1), (0, 0, 255), 3)
                    cvzone.putTextRect(framepeople, f'{id} Exiting', (cx, cy), 2, 2)
                    cv2.circle(framepeople, (cx, cy), 5, (0, 0, 255), -1)
                    if id not in counter2:
                        counter2.append(id)
            elif current_area == 'overlap':  # Nếu đang ở vùng chồng lấp
                if in_area1 and not in_area2:  # Rời khỏi overlap vào area1
                    state[id] = 'area1'
                elif in_area2 and not in_area1:  # Rời khỏi overlap vào area2
                    state[id] = 'area2'            
     
        alert_delay = 10
        lat=decimal_numbers[0]
        long=decimal_numbers[1]
        check=0
        if(lat==10.8595346 and long==106.8060503) :
                cv2.imwrite("hocsinh1.png", cv2.resize(framepeople, dsize=None, fx=1, fy=1))
                sendtele3(names)
                now=datetime.datetime.now()
                if last_alert_time is None or (now - last_alert_time).total_seconds() >= alert_delay:
                    last_alert_time = now
                    
                    check+=1
                    if(len(counter1)!=len(counter2))and check>1:
                            cv2.imwrite("alert.png", cv2.resize(framepeople, dsize=None, fx=1, fy=1))
                            sendtele() 
                            print("Send succes")             
            

        cv2.polylines(framepeople,[np.array(area1,np.int32)],True,(0,255,0),2) 
        cv2.polylines(framepeople,[np.array(area2,np.int32)],True,(0,0,255),2) 

        Enter=len(counter1)
        Exit=len(counter2)
        cvzone.putTextRect(framepeople,f'ENTER:{Enter}',(50,60),2,2)
        cvzone.putTextRect(framepeople,f'EXIT:{Exit}',(50,130),2,2)


        cv2.imshow('RGB', framepeople)
        cv2.setMouseCallback("RGB",RGB)
    #    time.sleep(0.1)
        if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
            break

    # Release the video capture and close windows
    video_capture.release()
    cv2.destroyAllWindows()


#điểm danh
MINSIZE = 20
THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.709
existing_names = []
attandanced_list=[]
INPUT_IMAGE_SIZE = 160
CLASSIFIER_PATH = 'E:/2025/khkt/khkt/Models/facemodel.pkl'
FACENET_MODEL_PATH = 'E:/2025/khkt/khkt/Models/20180402-114759.pb'
model_yolo = YOLO("E:/2025/khkt/khkt/Models/yolov8n-face.pt")
area3=[(446, 112),(443, 253),(639, 252),(629, 89)]
area4=[(597, 203),(591, 131),(483, 135),(490, 213)]

   
       
      
def attendance():
    with open(CLASSIFIER_PATH, 'rb') as file:
        model, class_names = pickle.load(file)
        print("Custom Classifier, Successfully loaded")
        print("classname=", class_names)
    with tf.Graph().as_default():
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():
            print('Loading feature extraction model')
            facenet.load_model(FACENET_MODEL_PATH)
            
            # Lấy các tensor đầu vào và đầu ra từ mô hình
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            people_detected = set()
            person_detected = collections.Counter()

    cap=cv2.VideoCapture("E:/2025/khkt/khkt/1.mp4")
    while True:     
        ret, frame= cap.read()
        frame=cv2.resize(frame,(1028,500))
        results = model_yolo(frame)[0]
        for result in results:
            boxes = result.cpu().boxes.numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                centroid=((x1+x2)//2,(y1+y2)//2)
                centroidx=(x1+x2)//2
                centroidy=(y1+y2)//2
                cv2.circle(frame, centroid, 5, (255,0,0), -1)
                cropped = frame[y1:y2, x1:x2]
                cropped = imutils.resize(cropped, width=450)
                result4=cv2.pointPolygonTest(np.array(area3,np.int32),((centroidx,centroidy)),False)
                # print ("result4",result4)
                if result4>=0:
                    result5=cv2.pointPolygonTest(np.array(area4,np.int32),((centroidx,centroidy)),False)
                    # print("result5",result5)
                    if result5>=0:    
                # Tiền xử lý ảnh trước khi trích xuất đặc trưng
                        scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
                        scaled = facenet.prewhiten(scaled)
                        scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                        feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                        emb_array = sess.run(embeddings, feed_dict=feed_dict)

                        # Dự đoán và tính xác suất
                        predictions = model.predict_proba(emb_array)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                        best_name = class_names[best_class_indices[0]]
                        print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))
                        print("len ", len(attandanced_list))
                        if len(attandanced_list)<3:
                            pro = 0.869
                        else:
                            pro = 0.82
                        if best_class_probabilities > pro:
                            print ("pro ",pro)
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            name = class_names[best_class_indices[0]]
                            if name not in attandanced_list:
                                cv2.imwrite(f"E:/2025/khkt/khkt/examples/{name}.png", cv2.resize(frame, dsize=None, fx=1, fy=1))
                                attandanced_list.append(name)
                                
                            current_date = datetime.datetime.now().strftime("%d-%m-%Y")
                            file_path = f'E:/2025/khkt/khkt/examples/di vao_{current_date}.csv'
                     

                            

                            # Kiểm tra nếu tên đã tồn tại
                            if name in existing_names:
                                print(f"The name '{name}' already exists in the file. No new entry added.")
                            else:
                                existing_names.append(name)
                                # Mở file ở chế độ thêm và ghi dữ liệu mới
                                with open(file_path, mode='a', newline='',encoding='utf-8') as file:
                                    writer = csv.writer(file)
                                    # Ghi tiêu đề nếu file chưa tồn tại
                                    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                                        writer.writerow(["Name", "Time"])
                                    # Ghi dữ liệu mới
                                    writer.writerow([name, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                                    print("Write success")
                            if name in names:
                                    names.remove(name)
                            cv2.putText(frame, name, (x1, y1 + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), thickness=1, lineType=2)       
        cv2.polylines(frame,[np.array(area3,np.int32)],True,(0,0,255),2) 
        cv2.polylines(frame,[np.array(area4,np.int32)],True,(0,0,255),2) 
        
        cv2.imshow("cropped", frame)
        cv2.setMouseCallback("cropped",RGB)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break            
countpeople_thread=threading.Thread(target=countpeople)     
attandance_thread=threading.Thread(target=attendance) 
gps_thread=threading.Thread(target=gps)
countpeople_thread.start()
# attandance_thread.start()
# gps_thread.start()
