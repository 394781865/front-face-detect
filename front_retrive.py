# -*- coding:utf-8 -*-
from flask import Flask,render_template,request
import face
from Queue import Queue
import threading
import numpy as np
import math
import base64
import os
from cStringIO import StringIO
import demjson
import xml.etree.cElementTree as et
import cv2
from scipy import misc

app = Flask(__name__)
#网页可视化
#@app.route('/')
#def index():
     #return render_template("index.html")
#创建两个队列   
q1 = Queue()
q2 = Queue()
       
#主函数
@app.route('/predict/', methods=['GET','POST'])
def predict():
    #获取json数据
    recv = demjson.decode(request.get_data())
    base64Data = recv["base64Data"]
    image_data = base64.b64decode(base64Data)
    image_data = StringIO(image_data)
 
    try:
        #image = cv2.imread(image_data,cv2.IMREAD_COLOR)
        #image = image[...,::-1]
        image = misc.imread(image_data)
    except:
        print "提示：读取发生异常！"
        data = {"isResult":"false","resultNum":"0"}
        data = demjson.encode(data)
        q2.put(data)
    else:
        q1.put(image)
        print("~~~~~~~~~~~数据已经加入队列q1~~~~~~~~~~")
    return q2.get()
  
#客户端函数    
def client():
    tree=et.parse("config/conf.xml")
    root=tree.getroot()
    port=root.find('port').text
    host=root.find('myhost').text
    port = int(os.environ.get("PORT", port))
    app.run(host=host, port=port)
                        
if __name__ == '__main__':
    #把人脸识别模型预先载入内存
    face_detection = face.Detection()
    #建立客户端连接线程
    parseImage_thread = threading.Thread(target=client)
    parseImage_thread.start()
 
    while True:
       while not q1.empty():
          value = q1.get() 
          print "从队列q1中取出数据"
          try:    
              faces = face_detection.find_faces(value)  
          except:
              print "提示：识别发生异常！"
              data = {"isResult":"false","resultNum":"0"}
              data = demjson.encode(data)
              q2.put(data)
          else:            
              if faces:
                 base64_dic = []
                 num = len(faces)
                 for one_face in faces:
                     bb_0 = one_face.bounding_box[0]
                     bb_1 = one_face.bounding_box[1]
                     bb_2 = one_face.bounding_box[2]
                     bb_3 = one_face.bounding_box[3] 
 
                     euler = one_face.euler             
                     if (np.abs(euler[0]) < 30 and np.abs(euler[1]) > 150):
                        data = {"isResult":"true","resultNum":str(num)}
                     else:
                        data = {"isResult":"false","resultNum":str(num)}

                     data = demjson.encode(data)
                 q2.put(data)           
                 print "检测数据已经放入队列q2-1"
              else:        
                     data = {"isResult":"false","resultNum":"0"}    
                     data = demjson.encode(data)  
                     q2.put(data)                   
                     print "检测数据已经放入队列q2-2"

