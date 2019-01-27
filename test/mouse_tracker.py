# -*- coding=utf-8 -*-
#!/usr/bin/python

# ref:https://blog.csdn.net/zuliang001/article/details/80912910

import cv2
import numpy as np
import sys
sys.path.append("..")
from filters import kalmanfilter as kf
# 创建一个大小800*800的空帧
frame = np.zeros((800, 800, 3), np.uint8)
# 初始化测量坐标和鼠标运动预测的数组
last_measurement = current_measurement = np.array((2, 1), np.float32)
last_predicition = current_prediction = np.zeros((2, 1), np.float32)

'''
    mousemove()函数在这里的作用就是传递X,Y的坐标值，便于对轨迹进行卡尔曼滤波
'''

def mousemove(event, x, y, s, p):
    # 定义全局变量
    global frame, current_measurement, last_measurement, current_prediction, last_prediction
    # 初始化
    last_measurement = current_measurement
    last_prediction = current_prediction
    # 传递当前测量坐标值
    current_measurement = np.array([[np.float32(x)], [np.float32(y)]])
    # 用来修正卡尔曼滤波的预测结果
    kalman.correct(np.mat(current_measurement))  # 用当前测量来校正卡尔曼滤波器
    # 调用kalman这个类的predict方法得到状态的预测值矩阵，用来估算目标位置
    u = np.mat(np.zeros((4,1)))
    current_prediction = kalman.predict(u)
    # 上一次测量值
    lmx, lmy = last_measurement[0], last_measurement[1]
    # 当前测量值
    cmx, cmy = current_measurement[0], current_measurement[1]
    # 上一次预测值
    lpx, lpy = last_prediction[0], last_prediction[1]
    # 当前预测值
    cpx, cpy = current_prediction[0], current_prediction[1]
    # 绘制测量值轨迹（绿色）
    cv2.line(frame, (lmx, lmy), (cmx, cmy), (0, 100, 0))
    # 绘制预测值轨迹（红色）
    cv2.line(frame, (lpx, lpy), (cpx, cpy), (0, 0, 200))



cv2.namedWindow("kalman_tracker")
# 调用函数处理鼠标事件，具体事件必须由回调函数的第一个参数来处理，该参数确定触发事件的类型（点击和移动）
cv2.setMouseCallback("kalman_tracker", mousemove)

kalman = kf.KalmanFilter(4,2)

# 设置测量矩阵
measurement_matrix = np.mat(np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32))
kalman.setMeasurementMatrix(measurement_matrix)

# 设置转移矩阵
transition_matrix = np.mat(np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32))
kalman.setTransitionMatrix(transition_matrix)

# 设置过程噪声协方差矩阵
process_noise_cov = np.mat(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03)
kalman.setProcessNoiseCov(process_noise_cov)

# 转换
measurement_noise_cov = np.mat(np.array([[1, 0], [0, 1]], np.float32) * 0.1)
kalman.setMeasurementNoisyCov(measurement_noise_cov)

kalman.setControlMatrix(np.mat(np.identity(4)))

while True:
    cv2.imshow("kalman_tracker", frame)
    if (cv2.waitKey(30) & 0xff) == 27:
        break

