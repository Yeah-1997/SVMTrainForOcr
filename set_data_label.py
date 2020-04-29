#制作训练集
import numpy as np
import cv2
from sklearn import svm
import matplotlib.colors
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score
import os
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from time import time
from sklearn.decomposition import PCA
from sklearn.externals import joblib

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.bmp','.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])
def pooling (img):
    he,we = img.shape
    h = int(he/2)
    w = int(we/2)
    img_new = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            img_new[i,j] = np.max(img[i*2:i*2+2,j*2:j*2+2])
    img_new = img_new.astype('uint8')
    return img_new

if __name__ == "__main__":
    tezhneg = 30
    pca = PCA(n_components=tezhneg, whiten=True, random_state=0)

#做出label

    y=np.array([],dtype=np.uint8)#标签
    #做出特征向量
    # filen = 'G:/我的地盘/毕设用/AA毕设\'s/训练集图片madebyhc/'#训练集图片所在的位置
    filen = 'G:/我的地盘/毕设用/AA毕设\'s/训练集图片madebycv/'#训练集图片所在的位置
    i = 0
    x =np.array([],dtype=np.uint8)#特征向量
    while i<12:  #遍历所有类别
        filename = filen+ str(i)
        image_filenames = [os.path.join(filename, x) for x in os.listdir(filename) if is_image_file(x)]  #得到每个类别下图片路径
        for picadd in image_filenames:   #遍历每类的图片
            img = cv2.imdecode(np.fromfile(picadd,dtype=np.uint8),-1)
            #提取特征向量，先中值滤波
            img_med  = cv2.medianBlur(img,3)
            # xm = pca.fit_transform(img_med)#主成分提取
            # img_pool = pooling(img_med)#池化
            x1 = img_med.reshape(1,-1)
            y = np.append(y,i)#打标签
            if len(x)==0:
                x = x1
            else:
                x = np.concatenate((x,x1),axis=0)
        i=i+1 
    y = y.reshape(-1,1)
    xfit = pca.fit_transform(x)#主成分提取
    print('数据维度：'),print(xfit.shape)
    print('标签维度：'),print(y.shape)
    print('Save data......')
    np.save('tarin_data',xfit)
    np.save('train_label',y)
    joblib.dump(pca,'pca_model')
    print('done')
    """ clf = svm.SVC(C=0.001, kernel='rbf')
    clf.fit(x, y.ravel())


    i=0
    re=[]
    while i<55:
        filename = filen+ str(i)+'.bmp'
        img1 = cv2.imdecode(np.fromfile(filename,dtype=np.uint8),-1)
        img1=cv2.resize(img1,(30,30))
        cv2.imshow('0',img1)
        cv2.waitKey(0)
        x2=img1.reshape(1,-1)
        re.append(clas[clf.predict(x2)])
        i=i+1

    print(np.array(re).reshape(-1,5))


    """



