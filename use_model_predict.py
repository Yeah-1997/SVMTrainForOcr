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
from set_data_label import is_image_file,pooling
from sklearn.decomposition import PCA


matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
print('加载SVM模型......\n')
svm_model = joblib.load('svm_model')
print('加载pca模型......\n')
pca = joblib.load('pca_model')
print('加载完成\n')
clas=np.array(['A','B','0','1','2','3','4','5','6','7','8','9']) 
filename = 'G:/我的地盘/毕设用/AA毕设\'s/测试集图片/'
image_filenames = [os.path.join(filename, x) for x in os.listdir(filename) if is_image_file(x)]  #得到每个图片路径

print('分类中......\n')
#画测试图片
plt.figure(figsize=(15, 10), facecolor='w')
for index,picadd in  enumerate(image_filenames):   #遍历每类的图片

    img = cv2.imdecode(np.fromfile(picadd,dtype=np.uint8),-1)#读取
    if img.ndim==3:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img,(30,30),interpolation=cv2.INTER_LINEAR)#归一化
    img_med  = cv2.medianBlur(img,3)#中值滤波
    # img_pool = pooling(img_med)#池化提取特征向量
    x1 = img_med.reshape(1,-1)
    y =  svm_model.predict(pca.transform(x1))#主成分分析后的向量送进去预测
    #y =  svm_model.predict(x1)
    plt.subplot(5,4, index + 1)
    plt.imshow(255-img, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(u'分为：%s' % clas[y][0])
plt.suptitle(u'分类结果', fontsize=20)
plt.tight_layout()
print('分类完成')
plt.show()
