#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import tensorflow as tf
import numpy as np
import math
import pdb
import dlib
import math
import os
# from deepgaze.head_pose_estimation import CnnHeadPoseEstimator

sess = tf.Session() #Launch the graph in a session.
# my_head_pose_estimator = CnnHeadPoseEstimator(sess) #Head pose estimation object
# my_head_pose_estimator.load_yaw_variables(os.path.realpath("etc/tensorflow/head_pose/yaw/cnn_cccdd_30k"))

detector = dlib.get_frontal_face_detector()
#detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat.bz2")
###各種設定（半円のどのへんを使うかはgetPoints内で設定してる）
setHeight = int(1440/2) #取り込み画像解像度(X)
setWidth = int(1440/2) #取り込み画像解像度(Y)
picCenter=(int(setWidth/2),int(setHeight/2)) #半円ミラーのセンター
valDiv=12 #一周の分割数
segDeg = 360 / valDiv #１セグメントの角度
valRad = 350 #透視変換する際のセンターからの半径
retWidth = 1500 #出力画像の解像度(X)
retHeight = valRad #出力画像の解像度(Y)
a=29.
b=40.
c=math.sqrt(a*a+b*b)
f=100.

def getPerspectiveImage(img, pts1,  pts2,width=50,height=300):
    M = cv2.getPerspectiveTransform(np.array(pts1).astype(np.float32), np.array(pts2).astype(np.float32))  # 透視変換行列を作成。
    return cv2.warpPerspective(img, M, (width, height))  # 透視変換行列を使って切り抜き。

def getdstPoint(x,y,deg):
    z=math.sqrt(x*x+y*y)*math.tan(math.radians(deg))
    t = (a*a*f)/(-(b*b+c)*z+2*b*c*abs(x))
    return (int(t*x)+picCenter[0], int(t*y)+picCenter[1])

def getPoints(Deg1,Deg2,R,C,width=50,height=300):
    S,L = 0.4,0.99 #センターから近い方の円周と遠い方の円周のそれぞれの割合

    x1 = int(R*L*math.cos(math.radians(Deg1)) + C[0])
    x2 = int(R*S*math.cos(math.radians(Deg1)) + C[0])
    x3 = int(R*S*math.cos(math.radians(Deg2)) + C[0])
    x4 = int(R*L*math.cos(math.radians(Deg2)) + C[0])

    y1 = int(R*L*math.sin(math.radians(Deg1)) + C[1])
    y2 = int(R*S*math.sin(math.radians(Deg1)) + C[1])
    y3 = int(R*S*math.sin(math.radians(Deg2)) + C[1])
    y4 = int(R*L*math.sin(math.radians(Deg2)) + C[1])
    pointxy = ((x1,y1),(x2,y2),(x3,y3),(x4,y4))
    pointXY = ((0,0),(0,height),(width,height),(width,0))

    return pointxy, pointXY
    
if __name__ == '__main__':
    
    valWidth = int(retWidth/valDiv)
    color = (0, 0, 255)
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, int(1440))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, int(1440))
   
    cam.set(cv2.CAP_PROP_FPS, 30)
    orig = cam.read()[1]
    print (orig.shape)




        
    valDeg1 = 1
    vP=[]
    hP=[]
    for i in range(1,valDiv+1):
        #分割した数だけ、透視変換処理を繰り替えす
        valDeg2 = segDeg * i
        #分割用のポイントを得る
        vPoints, hPoints =  getPoints(valDeg1,valDeg2,valRad,picCenter,valWidth,retHeight)
        vP.append(vPoints)
        hP.append(hPoints)
        valDeg1 = valDeg2

    while True:
        wmap=np.ones((500,500,3))*255
        cv2.circle(wmap,(250, 250),15,(0,0,0),-1)
        cv2.circle(wmap,(250, 250),100,(0,0,0),2)
        #orig = cv2.imread('img1.jpg')
        orig = cam.read()[1]
        orig = cv2.resize(orig,(int(setWidth),int(setHeight)))
        lines = orig.copy()
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cam.release()
            cv2.destroyAllWindows()

        #センターに丸をつける。
        cv2.circle(lines,picCenter,5,(0,255,0),-1)

        valDeg1 = 1
        sumWarped = None
        for i in range(0,valDiv):
            #分割した数だけ、透視変換処理を繰り替えす
            vPoints=vP[i]
            hPoints=hP[i]
        
            #分割ラインを描画
            cv2.line(lines,vPoints[2],vPoints[3],(0,255,0),3)
            
            #透視変換画像を得る
            warped = getPerspectiveImage(orig, vPoints,hPoints,valWidth,retHeight)
            
            #画像の結合
            if sumWarped is None: 
                sumWarped = warped.copy()
            else:
                sumWarped = cv2.hconcat([sumWarped, warped])
        
        sumWarped=cv2.flip(sumWarped,0)

        dets, scores, idx = detector.run(sumWarped, 0)
        for i, rect in enumerate(dets):
            deg = ((rect.left()+rect.right())/2.)/retWidth*360
            px=int(250+100*math.sin(math.radians(deg)))
            py=int(250+100*math.cos(math.radians(deg)))
            cv2.circle(wmap,(px, py),15,(0,0,255),-1)
            face=sumWarped[rect.top()-20:rect.bottom()+20,rect.left()-20:rect.right()+20,:]
            face = cv2.resize(face,(100,100))
            # yaw = my_head_pose_estimator.return_yaw(face)  # Evaluate the yaw angle using a CNN
            nosex=int(30*math.sin(math.radians(yaw[0][0][0]+deg)))
            nosey=int(30*math.cos(math.radians(yaw[0][0][0]+deg)))
            cv2.line(wmap,(px,py),(px-nosex,py-nosey),(0,255,0),3)
            print(yaw)
            cv2.imshow('face',face)
            cv2.rectangle(sumWarped, (rect.left()-20, rect.top()-20), (rect.right()+20, rect.bottom()+20), color, thickness=3)

        
        cv2.imshow('orig',lines)
        cv2.imshow('map',wmap)
        cv2.imshow('warp', sumWarped)
            
