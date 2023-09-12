import os
import cv2
import numpy as np

# 将XY方向光流图融合形成融合光流图

for root1,dirs1,files1 in os.walk(r"D:\\ARP"):
    for file in files1:
        if file[-8:-4] == '2330':
            continue
        for root2, dirs2, files2 in os.walk(r"D:\\ARP_Flow"):
            flow_x = cv2.imread("D:\\ARP_Flow_Add\\x_" + file)
            flow_y = cv2.imread("D:\\ARP_Flow_Add\\y_" + file)
            b2, g2, r2 = cv2.split(flow_x)
            b3, g3, r3 = cv2.split(flow_y)
        B = b2+b3
        G = g2+g3
        R = r2+r3
        img_merge = cv2.merge([B,G,R])
        cv2.imwrite('D:\\ARP_merge_flow\\'+file,img_merge)
        print(file)