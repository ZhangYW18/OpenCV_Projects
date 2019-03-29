import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import pandas as pd
import scipy.io as sio
import cv2
from PIL import Image



def build_density_map(density_map, output_dir, fname):
    # 需要加载map.mat
    map = sio.loadmat('./map.mat')
    map = map['c']
    map = map[::-1]
    density_map = 255*density_map/np.max(density_map)
    new_density = np.ones([density_map.shape[0], density_map.shape[1], 3], dtype=np.float)

    for i in range(density_map.shape[0]):
        for j in range(density_map.shape[1]):
            new_density[i][j] = map[int(density_map[i][j])]
    new_density = np.array(new_density*255)
    cv2.imwrite(os.path.join(output_dir, fname), new_density)


if __name__ == '__main__':   #import时忽略
    gt_path = './data/den/'
    output_dir = './data/density_map1/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    gt_files = [filename for filename in os.listdir(gt_path) if os.path.isfile(os.path.join(gt_path, filename))]
    for gt_file in gt_files:
        output_name = os.path.splitext(gt_file)[0] + '.png'
        now_path=os.path.join(gt_path, gt_file)
        den=Image.open(now_path)
        den=den.convert('L')
        den=den.getdata()
        den=np.matrix(den,dtype='float')
   #     den = pd.read_csv(os.path.join(gt_path, gt_file), sep=',', header=None).as_matrix()
        build_density_map(den, output_dir, fname=output_name)
