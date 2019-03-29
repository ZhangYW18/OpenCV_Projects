# Python 3.7.0, OpenCV 3.4.3
# Edited by VampireWeekend, 2018/10/13
import cv2
import numpy as np
import scipy.io as sio
import os
#导入表格
mp = sio.loadmat('./map.mat')
mp = mp['c']
mp = mp[::-1]
#创建路径
output_dir = './my_data/density_map/'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
#导入数据并处理
gt_path = './my_data/original'
gt_files = [filename for filename in os.listdir(gt_path) if os.path.isfile(os.path.join(gt_path, filename))]
for gt_file in gt_files:
    nowImg=cv2.imread(gt_path+'/'+gt_file)
    nowImg=cv2.cvtColor(nowImg,cv2.COLOR_BGR2GRAY);   #转为灰度图
    n=nowImg.shape[0]
    m=nowImg.shape[1]
    den_map = np.ones([n,m,3], dtype=np.float)
    for i in range(n):
        for j in range(m):
            pixel=nowImg[i][j]
            den_map[i][j]=mp[int(pixel)]*255
            den_map[i][j]=[int(x) for x in den_map[i][j]]
    cv2.imwrite(output_dir+gt_file,den_map)
