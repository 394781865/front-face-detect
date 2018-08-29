#coding=utf-8
import urllib2
import json
from socket import *
import os,time
from Queue import Queue
import demjson
import base64
import cv2

def http_post():  
        url = "http://192.168.1.186:5000/predict/" 
 
        f = open("/home/ubuntu/Additive-Margin-Softmax-master/dataset/raw/b/b_19.jpg")
        base64_data = base64.b64encode(f.read())
 
        data = {"base64Data":str(base64_data)}
        #postData = {"fileName":"0.png","base64Data":str("base64_data"),"X":"-0.958637","Y":"0.00766414455","Z":"-0.1388637"}
        postData = demjson.encode(data)
 
        req = urllib2.Request(url, postData)  
        req.add_header('Content-Type', 'application/json')  
        response = urllib2.urlopen(req)  
        result = json.loads(response.read())  
        print result    
 
http_post()
