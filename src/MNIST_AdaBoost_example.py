from matplotlib import pyplot
import cv2
import numpy as np
import struct
import time

time_start=time.time()


size = 20
N_feature = size * size


def cut_and_resize(image):
    x,y,w,h = cv2.boundingRect(image)
    #cut off space area
    img1 = image[y:(y+h), x:(x+w)]
    #resize image to 20*20
    img2 = cv2.resize(img1, (size,size))
    #flatten to vector
    vector = img2.flatten()
    return vector

def read_image(file_name):
    #先用二进制方式把文件都读进来
    file_handle=open(file_name,"rb")  #以二进制打开文档
    file_content=file_handle.read()   #读取到缓冲区中
    offset=0
    head = struct.unpack_from('>IIII', file_content, offset)  # 取前4个整数，返回一个元组
    offset += struct.calcsize('>IIII')
    imgNum = head[1]  #图片数
    rows = head[2]   #宽度
    cols = head[3]  #高度
 
    images=np.empty((imgNum , 28,28), dtype='uint8')#empty，是它所常见的数组内的所有元素均为空，没有实际意义，它是创建数组最快的方法
    image_size=rows*cols#单个图片的大小
    fmt='>' + str(image_size) + 'B'#单个图片的format
 
    for i in range(imgNum):
        images[i] = np.array(struct.unpack_from(fmt, file_content, offset)).reshape((rows, cols))
        offset += struct.calcsize(fmt)
    return images

def read_label(file_name):
    file_handle = open(file_name, "rb")  # 以二进制打开文档
    file_content = file_handle.read()  # 读取到缓冲区中
 
    head = struct.unpack_from('>II', file_content, 0)  # 取前2个整数，返回一个元组
    offset = struct.calcsize('>II')
 
    labelNum = head[1]  # label数
    # print(labelNum)
    bitsString = '>' + str(labelNum) + 'B'  # fmt格式：'>47040000B'
    label = struct.unpack_from(bitsString, file_content, offset)  # 取data数据，返回一个元组
    return np.array(label)

def loadDataSet():
    #mnist
    train_x_filename="data/data/MNIST/train-images-idx3-ubyte"
    train_y_filename="data/data/MNIST/train-labels-idx1-ubyte"
    test_x_filename="data/data/MNIST/t10k-images-idx3-ubyte"
    test_y_filename="data/data/MNIST/t10k-labels-idx1-ubyte"
 
    train_x=read_image(train_x_filename)#60000*784 的矩阵
    train_y=read_label(train_y_filename)#60000*1的矩阵
    test_x=read_image(test_x_filename)#10000*784
    test_y=read_label(test_y_filename)#10000*1
 
    return train_x, test_x, train_y, test_y

#load data
if __name__ == "__main__":
    print("Loading data...")
    train_X, test_X, train_y, test_y=loadDataSet()
    #(train_X, train_y), (test_X, test_y) = mnist.load_data()
    N_train = train_X.shape[0]
    N_test = test_X.shape[0]
    #cut and resize
    Train_X = np.zeros((N_train, N_feature))
    Test_X  = np.zeros((N_test, N_feature))
    for i in range(N_train):
        Train_X[i] = cut_and_resize(train_X[i])
    for i in range(N_test):
        Test_X[i]  = cut_and_resize(test_X[i])
    print("Complete!")

    #calculate frequencies
    freq = np.zeros(10)
    for i in range(10):
        freq[i] = np.mean(train_y == i)

    #calculate means and variances of each feature (use Gaussian model)
    print("Calculating means and variances...")
    means = np.zeros((10, N_feature))
    variances = np.zeros((10, N_feature))
    label = np.zeros((10, Train_X.shape[0]))

    for i in range(10):
        label[i] = train_y == i

    iter_num = 10
    d = np.ones(N_train) / N_train
    train_predict = np.zeros((10, N_train))
    train_predict_y = np.zeros(N_train)
    for i in range(iter_num):
        print("iter "+str(i))
        weighted_X = np.multiply(np.mat(d).T,Train_X)
        weighted_X_2 = np.multiply(np.mat(d).T, np.multiply(Train_X, Train_X))
        means = label @ weighted_X
        variances = label @ weighted_X_2 - np.multiply(means, means)
        for j in range(10):
            train_predict[j] = np.array(np.log(freq[j]) - np.sum(np.multiply(Train_X-means[j], Train_X-means[j]) / (2*np.multiply(variances[j], variances[j])), axis = 1))[:,0]
        for j in range(N_train):
            train_predict_y[j] = np.where(train_predict[:,j] == np.max(train_predict[:,j]))[0][0]
        error_rate = 1 - (train_predict_y == train_y).mean()
        print("error rate: "+str(error_rate))
        


    """n = 0
    #predict
    predict = np.zeros(10)
    for i in range(10):
        predict[i] = np.log(freq[i]) - np.sum(np.multiply(Test_X[n]-means[i], Test_X[n]-means[i]) / (2*np.multiply(variances[i], variances[i])))
    print(str(np.where(predict == np.max(predict))[0][0]))
    print(str(predict)) """

"""     #predict
    print("Predicting...")
    predict = np.zeros((10, N_test))
    predict_y = np.zeros(N_test)

    for i in range(10):
        predict[i] = np.log(freq[i]) - np.sum(np.multiply(Test_X-means[i], Test_X-means[i]) / (2*np.multiply(variances[i], variances[i])), axis = 1)   
    for i in range(N_test):
        predict_y[i] = np.where(predict[:,i] == np.max(predict[:,i]))[0][0]
    print("Complete!")

    accuracy = np.mean(predict_y == test_y)
    print("Accuracy: "+str(accuracy)) """

time_end=time.time()
print('total time:',time_end-time_start, "seconds")

""" print(str(np.where(predict == np.max(predict))[0][0]))
print(str(predict))

pyplot.subplot(1,1,1)
pyplot.imshow(test_X[n], cmap=pyplot.get_cmap('gray'))
pyplot.show()

#shape of dataset
print('X_train: ' + str(Train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(Test_X.shape))
print('Y_test:  '  + str(test_y.shape)) """
 
""" #plotting
from matplotlib import pyplot
j = 0
for k in range(10):
 for i in range(9):  
  pyplot.subplot(3,3,1+i)
  while train_y[j] != k:
   j = j + 1
  pyplot.imshow(Train_X[j], cmap=pyplot.get_cmap('gray'))
  j = j+1
 pyplot.show() """