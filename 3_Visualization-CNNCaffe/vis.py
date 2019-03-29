import numpy as np
import matplotlib.pyplot as plt
import os
import caffe
import sys
import pickle
import cv2

caffe_root = '/home/jcole/Git/caffe/'

deployPrototxt =  '/home/jcole/Git/caffe/models/bvlc_alexnet/deploy.prototxt'
modelFile = './bvlc_reference_caffenet.caffemodel'
meanFile = 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
#imageListFile = '/home/chenjie/DataSet/CompCars/data/train_test_split/classification/test_model431_label_start0.txt'
#imageBasePath = '/home/chenjie/DataSet/CompCars/data/cropped_image'
#resultFile = 'PredictResult.txt'


def initilize():
    print 'initilize ... '
    sys.path.insert(0, caffe_root + 'python')
 #   caffe.set_mode_gpu()
 #   caffe.set_device(4)
    net = caffe.Net(deployPrototxt, modelFile,caffe.TEST)
    return net

def getNetDetails(image, net):
    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.load(caffe_root + meanFile ).mean(1).mean(1)) # mean pixel
    transformer.set_raw_scale('data', 255)
    # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))
    # the reference model has channels in BGR order instead of RGB
    # set net to batch size of 50
    net.blobs['data'].reshape(1,3,227,227)

    net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(image))
    out = net.forward()

    filters = net.params['conv1'][0].data
    with open('FirstLayerFilter.pickle','wb') as f:
       pickle.dump(filters,f)
    vis_square(filters.transpose(0, 2, 3, 1))

    feat = net.blobs['conv1'].data[0, :36]
    with open('FirstLayerOutput.pickle','wb') as f:
       pickle.dump(feat,f)
    vis_square(feat,padval=1)
    pool = net.blobs['pool1'].data[0,:36]
    with open('pool1.pickle','wb') as f:
       pickle.dump(pool,f)
    vis_square(pool,padval=1)


def vis_square(data, padsize=1, padval=0 ):
    data -= data.min()
    data /= data.max()

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    print data.shape
    plt.imshow(data)
    plt.show()

if __name__ == "__main__":
    net = initilize()
    testimage = './xx.jpg'
    getNetDetails(testimage, net)
