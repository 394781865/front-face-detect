#coding=utf-8
#多进程裁剪人脸
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
  
import cv2
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import facenet
import detect_face
import random
from time import sleep
import face_preprocess
from multiprocessing import Process, Queue
import time

q1 = Queue()
mtcnn_modelfile="/home/ubuntu/Additive/cbir/model_check_points"
gpu_memory_fraction = 0.2
 
def main(q1): 
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, mtcnn_modelfile)
      
    minsize = 20 
    threshold = [ 0.6, 0.7, 0.7 ]
    factor = 0.709 
     
    while 1:
       value = q1.get()
       img = value[0]
       output_filename = value[1]
 
       bounding_boxes, points = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
       nrof_faces = bounding_boxes.shape[0]

       if nrof_faces>0:
          detect_multiple_faces = False
          det = bounding_boxes[:,0:4]
          det_arr = []
          img_size = np.asarray(img.shape)[0:2]
          margin = 44
 
          if nrof_faces>1:
             if detect_multiple_faces:
                for i in range(nrof_faces):
                    det_arr.append(np.squeeze(det[i]))
             else:
                bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                img_center = img_size / 2
                offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0]])
                offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                index = np.argmax(bounding_box_size-offset_dist_squared*2.0)
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
                  
               if nrof_faces == 1: 
                  _landmark = points[:,0].reshape((2,5)).T
                  img_face = face_preprocess.preprocess(img,bb,_landmark,'112,112') 
                  
               else:
                  crop = img[bb[1]:bb[3], bb[0]:bb[2], :] 
                  bounding, po = detect_face.detect_face(crop, minsize, pnet, rnet, onet, threshold, factor)
                  num_faces = bounding.shape[0]
 
                  if num_faces > 0:
                     bound = bounding[:,0:4]
                     bindex = 0
                     if num_faces > 1:
                        bounding_box_size = (bound[:,2]-bound[:,0])*(bound[:,3]-bound[:,1])
                        img_center = img_size / 2
                        offsets = np.vstack([(bound[:,0]+bound[:,2])/2-img_center[1], (bound[:,1]+bound[:,3])/2-img_center[0]])
 
                        offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                        bindex = np.argmax(bounding_box_size-offset_dist_squared*2.0) 
 
                     _landmark = po[:, bindex].reshape((2,5)).T
                     img_face = face_preprocess.preprocess(crop,bounding,_landmark,'112,112') 
 
               filename_base, file_extension = os.path.splitext(output_filename)
               if detect_multiple_faces:
                  output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
               else:
                  output_filename_n = "{}{}".format(filename_base, file_extension)
               img_face = img_face[...,::-1]
               cv2.imwrite(output_filename_n, img_face)
               print (output_filename_n)             
            
         
  
if __name__ == '__main__':
    for i in range(4):
       pw = Process(target=main, args=(q1,))
       pw.start()
 
    input_dir = "/home/ubuntu/Additive/dataset/raw"
    output_dir = "/home/ubuntu/Additive/dataset/align"
 
    # Add a random key to the filename to allow alignment using multiple processes
    random_key = np.random.randint(0, high=99999)
    bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)
    dataset = facenet.get_dataset(input_dir)
    print('Creating networks and loading parameters')
 
    with open(bounding_boxes_filename, "w") as text_file:
        nrof_images_total = 0
        random.shuffle(dataset)
 
        for cls in dataset:
            output_class_dir = os.path.join(output_dir, cls.name)
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)
                random.shuffle(cls.image_paths)
 
            for image_path in cls.image_paths:
                nrof_images_total += 1
                filename = os.path.splitext(os.path.split(image_path)[1])[0]
                output_filename = os.path.join(output_class_dir, filename+'.jpg')
                if not os.path.exists(output_filename):
                    try:
                        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
                        img = img[...,::-1]
                        #img = misc.imread(image_path)
                    except (IOError, ValueError, IndexError) as e:
                        errorMessage = '{}: {}'.format(image_path, e)
                        print(errorMessage)
                    else:
                        if img.ndim<2:
                            print('Unable to align "%s"' % image_path)
                            continue
                        if img.ndim == 2:
                            img = facenet.to_rgb(img)
                        img = img[:,:,0:3]  
                        value = []
                        value.append(img)
                        value.append(output_filename)     
                        q1.put(value)
                        size = q1.qsize()
                        if size > 1000:
                           print("~~~~~~~~~~~~~~~~~~~~~~"+str(size))
                           time.sleep(2)
                             
    print('Total number of images: %d' % nrof_images_total)
