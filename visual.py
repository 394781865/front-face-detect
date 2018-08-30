#coding=utf-8
import argparse
import sys,os,glob 
import time
import cv2
import face,h5py
import numpy as np
from scipy import misc
import face_preprocess 
import copy

sub_dir_full = "/home/ubuntu/Additive/cbir/http_mtcnn/face/test_in"
save_dir = "/home/ubuntu/Additive/cbir/http_mtcnn/face/test_out"
 
def main(face_recognition):
  face_detection = face.Detection()

  sub_dirs = os.walk(sub_dir_full).next()[1]
  for sub_dir in sub_dirs:
     dir_full = os.path.join(sub_dir_full, sub_dir)
     img_basenames = os.listdir(dir_full) 

     for im_name in img_basenames:
        full_imname = os.path.join(dir_full,str(im_name))
        img = cv2.imread(full_imname, cv2.IMREAD_COLOR)
        frame = img[...,::-1]
        image = copy.copy(img)
        try:
           faces = face_detection.find_faces(frame) 
           for face in faces:
               face_bb = face.bounding_box.astype(int)       
               cv2.rectangle(image,(face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),(0, 255, 0), 2)
               cv2.putText(image,str(face.euler[0]), (face_bb[0], face_bb[3]),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),thickness=2, lineType=2) 
               cv2.imwrite(save_dir+'/'+str(im_name),image)
               print ('Yes!')  
        except:
           print("error!")  
 
if __name__ == '__main__':
   main()
