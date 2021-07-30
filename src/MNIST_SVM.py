import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
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
    #cv2.threshold(img2,50,1,cv2.cv.CV_THRESH_BINARY_INV,img2)
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
    img_num = head[1]  #图片数
    rows = head[2]   #宽度
    cols = head[3]  #高度
 
    images=np.empty((img_num , 28,28), dtype='uint8')#empty，是它所常见的数组内的所有元素均为空，没有实际意义，它是创建数组最快的方法
    image_size=rows*cols#单个图片的大小
    fmt='>' + str(image_size) + 'B'#单个图片的format
 
    for i in range(img_num):
        images[i] = np.array(struct.unpack_from(fmt, file_content, offset)).reshape((rows, cols))
        offset += struct.calcsize(fmt)
    return images

def read_label(file_name):
    file_handle = open(file_name, "rb")  # 以二进制打开文档
    file_content = file_handle.read()  # 读取到缓冲区中
 
    head = struct.unpack_from('>II', file_content, 0)  # 取前2个整数，返回一个元组
    offset = struct.calcsize('>II')
 
    label_num = head[1]  # label数
    # print(label_num)
    bit_str = '>' + str(label_num) + 'B'  # fmt格式：'>47040000B'
    label = struct.unpack_from(bit_str, file_content, offset)  # 取data数据，返回一个元组
    return np.array(label)

def load_data_set():
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

def clip(alpha, L, H):
    ''' 修建alpha的值到L和H之间.
    '''
    if alpha < L:
        return L
    elif alpha > H:
        return H
    else:
        return alpha

def select_j(i, m):
    ''' 在m中随机选择除了i之外剩余的数
    '''
    l = list(range(m))
    seq = l[: i] + l[i+1:]
    return random.choice(seq)



def smo(train_x, train_y, c, iter_num):
    n_train, n_feature = train_x.shape
    alphas = np.zeros(n_train)
    b = 0
    it = 0
    def f(x):
        "SVM分类器函数 y = w^Tx + b"
        # Kernel function vector.
        x = np.matrix(x).T
        ks = train_x@x
        # Predictive value.
        wx = np.matrix(alphas*train_y)*ks
        fx = wx + b
        return fx[0, 0]
    while it < iter_num:
        pair_changed = 0
        for i in range(n_train):
            a_i, x_i, y_i = alphas[i], train_x[i], train_y[i]
            fx_i = f(x_i)
            E_i = fx_i - y_i
            j = select_j(i, n_train)
            a_j, x_j, y_j = alphas[j], train_x[j], train_y[j]
            fx_j = f(x_j)
            E_j = fx_j - y_j
            K_ii, K_jj, K_ij = np.dot(x_i, x_i), np.dot(x_j, x_j), np.dot(x_i, x_j)
            eta = K_ii + K_jj - 2*K_ij
            if eta <= 0:
                print('WARNING  eta <= 0')
                continue
            # 获取更新的alpha对
            a_i_old, a_j_old = a_i, a_j
            a_j_new = a_j_old + y_j*(E_i - E_j)/eta
            # 对alpha进行修剪
            if y_i != y_j:
                L = max(0, a_j_old - a_i_old)
                H = min(c, c + a_j_old - a_i_old)
            else:
                L = max(0, a_i_old + a_j_old - c)
                H = min(c, a_j_old + a_i_old)
            a_j_new = clip(a_j_new, L, H)
            a_i_new = a_i_old + y_i*y_j*(a_j_old - a_j_new)
            if abs(a_j_new - a_j_old) < 0.00000001:
                #print('WARNING   alpha_j not moving enough')
                continue
            alphas[i], alphas[j] = a_i_new, a_j_new
            # 更新阈值b
            b_i = -E_i - y_i*K_ii*(a_i_new - a_i_old) - y_j*K_ij*(a_j_new - a_j_old) + b
            b_j = -E_j - y_i*K_ij*(a_i_new - a_i_old) - y_j*K_jj*(a_j_new - a_j_old) + b
            if 0 < a_i_new < c:
                b = b_i
            elif 0 < a_j_new < c:
                b = b_j
            else:
                b = (b_i + b_j)/2
            pair_changed += 1
            #print('INFO   iteration:{}  i:{}  pair_changed:{}'.format(it, i, pair_changed))
        if pair_changed == 0:
            it += 1
        else:
            it = 0
        #print('iteration number: {}'.format(it))
    return alphas, b

def get_w(alphas, dataset, labels):
    ''' 通过已知数据点和拉格朗日乘子获得分割超平面参数w
    '''
    alphas, dataset, labels = np.array(alphas), np.array(dataset), np.array(labels)
    yx = (labels.reshape(1, -1).T*np.ones(dataset.shape[1],dtype=int))*dataset
    w = np.dot(yx.T, alphas)
    return w.tolist()

if __name__ == "__main__":
    #load data
    print("Loading data...")
    train_X, test_X, train_y, test_y=load_data_set()
    N_train = train_X.shape[0]
    N_test = test_X.shape[0]
    #cut and resize
    Train_X = np.zeros((N_train, N_feature),dtype="uint8")
    Test_X  = np.zeros((N_test, N_feature),dtype="uint8")
    for i in range(N_train):
        Train_X[i] = cut_and_resize(train_X[i])

    for i in range(N_test):
        Test_X[i]  = cut_and_resize(test_X[i])

    print("Complete!")

    time_build_start = time.time()
    # calculating
    b = np.zeros((10, 10))
    w = np.zeros((10, 10, N_feature))
    for i in range(10):
        for j in range(i+1,10):
            print("Trainning classifier "+str(i)+str(j))
            t = ((train_y == i).astype(int) + (train_y==j).astype(int))
            m = np.multiply(Train_X, np.mat(t).T)
            index_array = np.zeros(N_train, dtype = bool)
            id_list=np.where((m.sum(axis=1))!=0)
            idlist=id_list[0]
            Train_X_ij = m[idlist]
            train_y_ij = train_y[idlist]
            train_y_ij = (train_y_ij == i).astype(int) - (train_y_ij == j).astype(int) 
            alphas, b[i][j] = smo(np.array(Train_X_ij)[:100,:], train_y_ij[:100], 0.6, 40)
            w[i][j] = np.multiply(alphas, train_y_ij[:100]) @ Train_X_ij[:100,:]


    time_build_end = time.time()

    # predicting
    predict = np.zeros((N_test, 10))
    predict_y = np.zeros(N_test)
    for k in range(N_test):
        for i in range(10):
            for j in range(i+1, 10):
                a = w[i][j] @ Test_X[k].T+b[i][j]
                #print(i,j,a)
                if a > 0:
                    predict[k][i] += 1
                else:
                    predict[k][j] += 1
        predict_y[k] = np.where(predict[k] == predict[k].max())[0][0]
    
    accuracy = (predict_y == test_y).mean()
    print(accuracy)


    time_end=time.time()

    print("Building model time: "+str(time_build_end-time_build_start)+" seconds")
    print("Predicting time: "+str(time_end-time_build_end)+" seconds")
    print('Total time:',time_end-time_start, "seconds")

""" t = ((test_y == i).astype(int) + (test_y==j).astype(int))
m = np.multiply(Test_X, np.mat(t).T)
index_array = np.zeros(N_train, dtype = bool)
id_list=np.where((m.sum(axis=1))!=0)
idlist=id_list[0]
Test_X_ij = m[idlist]
test_y_ij = test_y[idlist]
predict_y_ij = np.zeros(test_y_ij.shape[0])
for i in range(test_y_ij.shape[0]):
    if(w[0][1] @ Test_X_ij[i].T > 0):
        predict_y_ij[i] = 0
    else:
        predict_y_ij[i] = 1

accuracy = (predict_y_ij==test_y_ij).mean() """


"""     train_y_0 = (train_y == 0).astype(int) - (train_y != 0).astype(int)
    test_y_0 = test_y == 0
    alphas_0, b_0 = smo(Train_X[:100,:], train_y_0[:100], 0.6, 40)
    w_0 = (np.multiply(alphas_0, train_y_0[:100]) @ Train_X[:100,:])
    print("alphas"+str(alphas_0))
    print("b_0:"+str(b_0))
    print("w_0:"+str(w_0))
    "f(x) = w * x + b"
    i=0
    predict_y = np.zeros(N_test)
    for i in range(N_test):
        predict_y[i] = (w_0 @ Test_X[i].T + b_0 > 0).astype(int)
    accuracy = (predict_y == test_y_0).mean()
    print(accuracy) """