import numpy as np
import scipy.io as sio
import cv2
import os

path = 'D:\\OJ\\OpenCV_Projects\\2_crowd_density_map\\data\\test_data\\ground_truth\\GT_IMG_'
ipath = 'D:\\OJ\\OpenCV_Projects\\2_crowd_density_map\\data\\test_data\\images\\IMG_'
len = 182
R = 50
r = np.sqrt((R//2)**2*2)

mp = sio.loadmat('./map.mat')
mp = mp['c']
mp = mp[::-1]
out_path = './density_map/'
if not os.path.exists(out_path):
    os.mkdir(out_path)

for i in range(1,len+1):
    now_path = path+''+str(i)+'.mat'
    data = sio.loadmat(now_path)
    P = data['image_info']
    n = P[0][0][0][0][1][0][0]
    P = P[0][0][0][0][0]
    image_path = ipath+str(i) + '.jpg'
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
    N = img.shape[0]
    M = img.shape[1]
    den_map = np.zeros([N, M, 3], dtype=np.float)
    tot = np.zeros([N, M], dtype=np.float)
    for j in range(n):
        y = int(P[j][0])
        x = int(P[j][1])
        for X in range(x-R//2,x+R//2+1):
            if X < 0 or X >= N:
                continue
            for Y in range(y-R//2,y+R//2+1):
                if Y < 0 or Y >= M:
                    continue
                dis = np.sqrt((X-x) ** 2 + (Y-y) ** 2)
                add = np.exp(dis*(-4)/r)
    #            if j == 1:
    #                print(add)
                tot[X][Y] += add
    max_den=tot.max()
    for X in range(N):
        for Y in range(M):
            pixel = 255*tot[X][Y]/max_den
            den_map[X][Y] = mp[int(pixel)] * 255
            den_map[X][Y] = [int(ele) for ele in den_map[X][Y]]
    cv2.imwrite(out_path + 'density_map_' + str(i) + '.jpg', den_map)
    print(str(i)+'\\'+str(len))