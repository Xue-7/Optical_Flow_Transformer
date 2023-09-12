import os
import numpy as np
import cv2

# 获得每个气象因子相邻时间帧之间的XY光流图

bound = 15
for root, dirs, files in os.walk(r"D:\\ARP\\"):
    for i in range(len(files) - 1):
        file1 = files[i]
        file2 = files[i + 1]
        if file2[-8:-4] == '2330':
            continue
        # if file1[:-1] == '2330'
        frame1 = cv2.imread('D:\\ARP\\' + file1)
        frame2 = cv2.imread('D:\\ARP\\' + file2)
        hsv = np.zeros_like(frame1)
        prev = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        curr = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
        flow = TVL1.calc(prev, curr, None)
        # print(flow[:,:,0])
        assert flow.dtype == np.float32

        flow = (flow + bound) * (255.0 / (2 * bound))
        flow = np.round(flow).astype(int)
        flow[flow >= 255] = 255
        flow[flow <= 0] = 0

        # print(flow[:,:,0])

        cv2.imwrite('D:\\ARP_Flow\\x_' + file2, flow[:, :, 0])
        cv2.imwrite('D:\\ARP_Flow\\y_' + file2, flow[:, :, 1])
        print(file2)

for root, dirs, files in os.walk(r"D:\\Cloud\\"):
    for i in range(len(files) - 1):
        file1 = files[i]
        file2 = files[i + 1]
        if file2[-8:-4] == '2330':
            continue
        # if file1[:-1] == '2330'
        frame1 = cv2.imread('D:\\Cloud\\' + file1)
        frame2 = cv2.imread('D:\\Cloud\\' + file2)
        hsv = np.zeros_like(frame1)
        prev = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        curr = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
        flow = TVL1.calc(prev, curr, None)
        # print(flow[:,:,0])
        assert flow.dtype == np.float32

        flow = (flow + bound) * (255.0 / (2 * bound))
        flow = np.round(flow).astype(int)
        flow[flow >= 255] = 255
        flow[flow <= 0] = 0

        # print(flow[:,:,0])

        cv2.imwrite('D:\\Cloud_Flow\\x_' + file2, flow[:, :, 0])
        cv2.imwrite('D:\\Cloud_Flow\\y_' + file2, flow[:, :, 1])
        print(file2)

for root, dirs, files in os.walk(r"D:\\CLP\\"):
    for i in range(len(files) - 1):
        file1 = files[i]
        file2 = files[i + 1]
        if file2[-8:-4] == '2330':
            continue
        # if file1[:-1] == '2330'
        frame1 = cv2.imread('D:\\CLP\\' + file1)
        frame2 = cv2.imread('D:\\CLP\\' + file2)
        hsv = np.zeros_like(frame1)
        prev = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        curr = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
        flow = TVL1.calc(prev, curr, None)
        # print(flow[:,:,0])
        assert flow.dtype == np.float32

        flow = (flow + bound) * (255.0 / (2 * bound))
        flow = np.round(flow).astype(int)
        flow[flow >= 255] = 255
        flow[flow <= 0] = 0

        # print(flow[:,:,0])

        cv2.imwrite('D:\\CLP_Flow\\x_' + file2, flow[:, :, 0])
        cv2.imwrite('D:\\CLP_Flow\\y_' + file2, flow[:, :, 1])
        print(file2)

for root, dirs, files in os.walk(r"D:\\CLOT\\"):
    for i in range(len(files) - 1):
        file1 = files[i]
        file2 = files[i + 1]
        if file2[-8:-4] == '2330':
            continue
        # if file1[:-1] == '2330'
        frame1 = cv2.imread('D:\\CLOT\\' + file1)
        frame2 = cv2.imread('D:\\CLOT\\' + file2)
        hsv = np.zeros_like(frame1)
        prev = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        curr = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
        flow = TVL1.calc(prev, curr, None)
        # print(flow[:,:,0])
        assert flow.dtype == np.float32

        flow = (flow + bound) * (255.0 / (2 * bound))
        flow = np.round(flow).astype(int)
        flow[flow >= 255] = 255
        flow[flow <= 0] = 0

        # print(flow[:,:,0])

        cv2.imwrite('D:\\CLOT_Flow\\x_' + file2, flow[:, :, 0])
        cv2.imwrite('D:\\CLOT_Flow\\y_' + file2, flow[:, :, 1])
        print(file2)