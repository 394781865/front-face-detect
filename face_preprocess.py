import cv2
import numpy as np
from skimage import transform as trans
import math,time

def parse_lst_line(line):
  vec = line.strip().split("\t")
  assert len(vec)>=3
  aligned = int(vec[0])
  image_path = vec[1]
  label = int(vec[2])
  bbox = None
  landmark = None
  #print(vec)
  if len(vec)>3:
    bbox = np.zeros( (4,), dtype=np.int32)
    for i in xrange(3,7):
      bbox[i-3] = int(vec[i])
    landmark = None
    if len(vec)>7:
      _l = []
      for i in xrange(7,17):
        _l.append(float(vec[i]))
      landmark = np.array(_l).reshape( (2,5) ).T
  #print(aligned)
  return image_path, label, bbox, landmark, aligned
 
def read_image(img_path, **kwargs):
  mode = kwargs.get('mode', 'rgb')
  layout = kwargs.get('layout', 'HWC')
  if mode=='gray':
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
  else:
    #img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = img[...,::-1]
    #img = np.transpose(img, (2,0,1))
  return img
 
def preprocess(img,landmark):
 
    model_points = np.array([
      [-165.0, 170.0, -115.0],                        # Left eye left corner
      [165.0, 170.0, -115.0],                         # Right eye right corne
      [0.0, 0.0, 0.0],                                # Nose tip
      [-150.0, -150.0, -125.0],                       # Left Mouth corner
      [150.0, -150.0, -125.0]],dtype=np.float32)      # Right mouth corner        
  
    focal_length = img.shape[1]                          
    center = (img.shape[1]/2, img.shape[0]/2)
    camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = np.float32
                            )

    dst = landmark.astype(np.float32)
    dist_coeffs = np.zeros((4,1))
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, dst, camera_matrix, dist_coeffs)
    
    rotation3_3 = cv2.Rodrigues(rotation_vector)[0]

    q0 = np.sqrt(1 + rotation3_3[0][0] + rotation3_3[1][1] + rotation3_3[2][2]) / 2
    q1 = (rotation3_3[2][1] - rotation3_3[1][2]) / (4*q0) 
    q2 = (rotation3_3[0][2] - rotation3_3[2][0]) / (4*q0) 
    q3 = (rotation3_3[1][0] - rotation3_3[0][1]) / (4*q0) 

    yaw  = math.asin(2*(q0*q2 + q1*q3))*(180/math.pi)
    pitch= math.atan2(2*(q0*q1-q2*q3), q0*q0-q1*q1-q2*q2+q3*q3)*(180/math.pi)
    #roll = math.atan2(2*(q0*q3-q1*q2), q0*q0+q1*q1-q2*q2-q3*q3)*(180/math.pi)
    euler = [yaw,pitch]

    return euler

