import cv2
import numpy as np
from skimage import transform as trans
 
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
 
def preprocess(img,bb,landmark,image_size):
  M = None
  if len(image_size)>0:
    image_size = [int(x) for x in image_size.split(',')]
    if len(image_size)==1:
       image_size = [image_size[0], image_size[0]]
 
  if landmark is not None:
    src = np.array([
      [30.2946, 51.6963],
      [65.5318, 51.5014],
      [48.0252, 71.7366],
      [33.5493, 92.3655],
      [62.7299, 92.2041] ], dtype=np.float32 )

    if image_size[1]==112:
      src[:,0] += 8.0
    dst = landmark.astype(np.float32)
 
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2,:]
 
  if M is None:
     ret = img[bb[1]:bb[3],bb[0]:bb[2],:]
     if len(image_size)>0:
        ret = cv2.resize(ret, (image_size[1], image_size[0]))
     return ret
  else: 
     warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)
     return warped

