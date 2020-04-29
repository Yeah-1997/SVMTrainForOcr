
#训练SVM模型并保存
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
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score
from sklearn.externals import joblib
from time import time

if __name__ == "__main__":

    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    clas=np.array(['A','B','0','1','2','3','4','5','6','7','8','9']) 
    x = np.load('tarin_data.npy')
    print(x.shape)
    y = np.load('train_label.npy')
    print(y.shape)

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.6)
    c = 0.4
    gamma = 0.008
    clf = svm.SVC(C=c, kernel='rbf',gamma=gamma) #20主特征  c=0.3 gamma = 0.008halcon来的数据 20主特征c=0.45,gamma = 0.02 cv来的数据集 
    print('训练开始.....')
    print('c=',c,'\t','gamma=',gamma)
    clf.fit(x_train, y_train.ravel())
    print('训练结束.....')
    #print( clf.score(x_train, y_train)  )# 精度
    y_test_hat=clf.predict(x_test)
    print ('训练集准确率：', accuracy_score(y_train, clf.predict(x_train)))
    print ('测试集准确率：', accuracy_score(y_test,y_test_hat))
  #  print('召回率',recall_score(y_test,y_test_hat,average=None))
    #print (clf.score(x_test, y_test))
    y_test = y_test.ravel()
    
    img_test = x_test.reshape(-1,1,30)#测试集数据图片
    #画测试图片
    plt.figure(figsize=(10, 8), facecolor='w')
    for index,img  in enumerate(img_test):
        if index>=12:
            break
        plt.subplot(12, 1, index + 1)
        plt.axis('off')
        plt.imshow(255-img, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title(u'训练特征：%s' % clas[y_test[index]])
    plt.tight_layout()
    plt.show()

    #画分类错误的图片
    e_y = y_test_hat[y_test_hat!=y_test] #预测错误的值
    r_y= y_test[y_test_hat!=y_test]  #实际值

    error_img = img_test[(y_test_hat!=y_test)]
    plt.figure(figsize=(10, 8), facecolor='w')
    for index,img  in enumerate(error_img):
            if index>=4:
                break
            plt.subplot(2, 2, index + 1)
            plt.imshow(255-img, cmap=plt.cm.gray_r, interpolation='nearest')
            plt.axis('off')
            plt.title(u'错分为：%s，真实值：%s' % (clas[e_y[index]], clas[r_y[index]]))
    plt.tight_layout()
    plt.show()

    #保存模型
    joblib.dump(clf,'svm_model')
    print('模型保存完毕')



