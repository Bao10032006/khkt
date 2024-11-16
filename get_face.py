from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ultralytics import YOLO
import cv2
import imutils
import os

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import facenet
import align.detect_face
import random
from time import sleep
#Get face (595, 225),(490, 226
area3=[(446, 112),(443, 260),(639, 260),(629, 89)]
area4=[(587, 135),(596, 235),(492, 240),(487, 138)]
model_yolo = YOLO("Models/yolov10s-face.pt")
parent_dir= './DataSet/FaceData/raw/'
directory = "Tính"
path = os.path.join(parent_dir, directory) 
try:
    os.mkdir(path) 
except:
    pass

cap = cv2.VideoCapture("E:/2025/khkt/khkt/examples/4.mp4")
# cap=cv2.VideoCapture(0)
cnt = 0 
stt = False
check_out = False
result5 = -1
while True:
    _,frame = cap.read()
    frame=cv2.resize(frame,(1028,500))
    # frame = cv2.flip(frame, 1)
    results = model_yolo(frame)[0]
    for result in results:
        boxes = result.cpu().boxes.numpy()
        for box in (boxes):
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            cropped = frame[y1:y2,x1:x2]
            cropped = imutils.resize(cropped, width=450)
            centroid=((x1+x2)//2,(y1+y2)//2)
            centroidx=(x1+x2)//2
            centroidy=(y1+y2)//2
            cv2.circle(frame, centroid, 5, (255,0,0), -1)
            result4=cv2.pointPolygonTest(np.array(area3,np.int32),((centroidx,centroidy)),False)
            # print ("result4",result4)
            
            if result4>=0:
                result5=cv2.pointPolygonTest(np.array(area4,np.int32),((centroidx,centroidy)),False)
                # print("result5",result5)
                if result5>=0:  
                    cnt = cnt + 1
                    check_out = True
                    print(cnt)
                    cv2.imwrite('./DataSet/FaceData/raw/{}/{}.jpg'.format(directory,cnt),cropped)
            if result5 == -1 and check_out == True:
                stt = True
                break
            cv2.rectangle(frame,(x1, y1),(x2, y2), (0, 255, 0), 2)
        if stt:
            break
    cv2.waitKey(1)
    # if cv2.waitKey(50)& 0xFF == ord('s'):
        
        # if cnt ==30:
        #     break

    cv2.polylines(frame,[np.array(area3,np.int32)],True,(0,0,255),2) 
    cv2.polylines(frame,[np.array(area4,np.int32)],True,(0,0,255),2)     
    cv2.imshow("cam", frame)
    if stt: 
        break
cap.release()

input_dir = "Dataset/FaceData/raw"
output_dir = "Dataset/FaceData/processed"
image_size = 160
margin = 32
random_order = True
detect_multiple_faces = False
sleep(random.random())
output_dir = os.path.expanduser(output_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# Store some git revision info in a text file in the log directory
src_path,_ = os.path.split(os.path.realpath(__file__))
facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
dataset = facenet.get_dataset(input_dir)
print("dataset ", dataset)
print('Creating networks and loading parameters')

with tf.Graph().as_default():
    sess = tf.compat.v1.Session()
    with sess.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

# Add a random key to the filename to allow alignment using multiple processes
random_key = np.random.randint(0, high=99999)
bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)

with open(bounding_boxes_filename, "w", encoding="utf-8") as text_file:
    nrof_images_total = 0
    nrof_successfully_aligned = 0
    if random_order:
        random.shuffle(dataset)
    for cls in dataset:
        output_class_dir = os.path.join(output_dir, cls.name)
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)
            if random_order:
                random.shuffle(cls.image_paths)
        for image_path in cls.image_paths:
            nrof_images_total += 1
            filename = os.path.splitext(os.path.split(image_path)[1])[0]
            output_filename = os.path.join(output_class_dir, filename+'.png')
            print(image_path)
            if not os.path.exists(output_filename):
                try:
                    import imageio
                    img = imageio.imread(image_path)
                except (IOError, ValueError, IndexError) as e:
                    errorMessage = '{}: {}'.format(image_path, e)
                    print(errorMessage)
                else:
                    if img.ndim<2:
                        print('Unable to align "%s"' % image_path)
                        text_file.write('%s\n' % (output_filename))
                        continue
                    if img.ndim == 2:
                        img = facenet.to_rgb(img)
                    img = img[:,:,0:3]

                    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                    nrof_faces = bounding_boxes.shape[0]
                    if nrof_faces>0:
                        det = bounding_boxes[:,0:4]
                        det_arr = []
                        img_size = np.asarray(img.shape)[0:2]
                        if nrof_faces>1:
                            if detect_multiple_faces:
                                for i in range(nrof_faces):
                                    det_arr.append(np.squeeze(det[i]))
                            else:
                                bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                                img_center = img_size / 2
                                offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                                offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                                index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                                det_arr.append(det[index,:])
                        else:
                            det_arr.append(np.squeeze(det))

                        for i, det in enumerate(det_arr):
                            det = np.squeeze(det)
                            bb = np.zeros(4, dtype=np.int32)
                            bb[0] = np.maximum(det[0]-margin/2, 0)
                            bb[1] = np.maximum(det[1]-margin/2, 0)
                            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
                            bb[3] = np.minimum(det[3]+margin/2, img_size[0])
                            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                            from PIL import Image
                            cropped = Image.fromarray(cropped)
                            scaled = cropped.resize((image_size, image_size), Image.BILINEAR)
                            nrof_successfully_aligned += 1
                            filename_base, file_extension = os.path.splitext(output_filename)
                            if detect_multiple_faces:
                                output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                            else:
                                output_filename_n = "{}{}".format(filename_base, file_extension)
                            imageio.imwrite(output_filename_n, scaled)
                            text_file.write('%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))
                    else:
                        print('Unable to align "%s"' % image_path)
                        text_file.write('%s\n' % (output_filename))
# import os
# import sys
# import random
# import numpy as np
# import imageio
# from PIL import Image
# from ultralytics import YOLO
# import facenet  # Import facenet nếu cần cho các hàm phụ trợ như chuyển ảnh grayscale sang RGB

# # Cài đặt các thông số cơ bản
# input_dir = "Dataset/FaceData/raw"
# output_dir = "Dataset/FaceData/processed"
# image_size = 160
# margin = 32
# random_order = True
# detect_multiple_faces = False
# minsize = 20  # Kích thước tối thiểu của khuôn mặt
# threshold = 0.6  # Ngưỡng xác suất tối thiểu để chấp nhận phát hiện là khuôn mặt

# # Thiết lập thư mục đầu ra và ghi thông tin git
# output_dir = os.path.expanduser(output_dir)
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# src_path, _ = os.path.split(os.path.realpath(__file__))
# facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))

# # Tải dataset
# dataset = facenet.get_dataset(input_dir)
# print("dataset ", dataset)
# print('Creating networks and loading parameters')

# # Tải mô hình YOLOv8 cho phát hiện khuôn mặt
# model = YOLO('Models/yolov10s-face.pt')  # Sử dụng mô hình YOLOv8 đã huấn luyện cho khuôn mặt

# # Thêm một khóa ngẫu nhiên vào tên tệp để cho phép căn chỉnh bằng nhiều tiến trình
# random_key = np.random.randint(0, high=99999)
# bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)

# with open(bounding_boxes_filename, "w") as text_file:
#     nrof_images_total = 0
#     nrof_successfully_aligned = 0
#     if random_order:
#         random.shuffle(dataset)
#     for cls in dataset:
#         output_class_dir = os.path.join(output_dir, cls.name)
#         if not os.path.exists(output_class_dir):
#             os.makedirs(output_class_dir)
#         if random_order:
#             random.shuffle(cls.image_paths)
        
#         for image_path in cls.image_paths:
#             nrof_images_total += 1
#             filename = os.path.splitext(os.path.split(image_path)[1])[0]
#             output_filename = os.path.join(output_class_dir, filename + '.png')
#             print(image_path)
            
#             if not os.path.exists(output_filename):
#                 try:
#                     img = imageio.imread(image_path)
#                 except (IOError, ValueError, IndexError) as e:
#                     errorMessage = '{}: {}'.format(image_path, e)
#                     print(errorMessage)
#                 else:
#                     if img.ndim < 2:
#                         print('Không thể căn chỉnh "{}"'.format(image_path))
#                         text_file.write('%s\n' % output_filename)
#                         continue
                    
#                     if img.ndim == 2:  # Nếu ảnh là ảnh grayscale, chuyển sang RGB
#                         img = facenet.to_rgb(img)
#                     img = img[:, :, :3]  # Chỉ giữ lại ba kênh đầu tiên

#                     # Phát hiện khuôn mặt bằng YOLOv8
#                     results = model.predict(img, conf=threshold)
#                     faces = results[0].boxes  # Lấy các bounding box của khuôn mặt

#                     nrof_faces = len(faces)
#                     if nrof_faces > 0:
#                         det_arr = []
#                         img_size = np.asarray(img.shape)[0:2]

#                         if nrof_faces > 1:
#                             if detect_multiple_faces:
#                                 det_arr = faces
#                             else:
#                                 # Chọn khuôn mặt gần trung tâm ảnh nhất
#                                 img_center = img_size / 2
#                                 min_dist = float("inf")
#                                 selected_face = None
#                                 for face in faces:
#                                     box = face.xyxy[0]  # Lấy tọa độ (x1, y1, x2, y2)
#                                     center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
#                                     dist = np.sum((np.array(center) - img_center) ** 2)
#                                     if dist < min_dist:
#                                         min_dist = dist
#                                         selected_face = face
#                                 det_arr = [selected_face]
#                         else:
#                             det_arr = faces

#                         for i, face in enumerate(det_arr):
#                             box = face.xyxy[0].astype(np.int32)  # Lấy tọa độ (x1, y1, x2, y2)
#                             bb = np.zeros(4, dtype=np.int32)
#                             bb[0] = np.maximum(box[0] - margin // 2, 0)
#                             bb[1] = np.maximum(box[1] - margin // 2, 0)
#                             bb[2] = np.minimum(box[2] + margin // 2, img_size[1])
#                             bb[3] = np.minimum(box[3] + margin // 2, img_size[0])
                            
#                             cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
#                             cropped = Image.fromarray(cropped)
#                             scaled = cropped.resize((image_size, image_size), Image.BILINEAR)
                            
#                             nrof_successfully_aligned += 1
#                             filename_base, file_extension = os.path.splitext(output_filename)
#                             if detect_multiple_faces:
#                                 output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
#                             else:
#                                 output_filename_n = "{}{}".format(filename_base, file_extension)
                            
#                             imageio.imwrite(output_filename_n, np.array(scaled))
#                             text_file.write('%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))
#                     else:
#                         print('Không thể căn chỉnh "{}"'.format(image_path))
#                         text_file.write('%s\n' % output_filename)
print('Total number of images: %d' % nrof_images_total)
print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
