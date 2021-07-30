import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
import struct
import time
from sklearn.feature_extraction.text import CountVectorizer

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

def f(x, train_x, train_y, alphas, b):
        "SVM分类器函数 y = w^Tx + b"
        # Kernel function vector.
        x = np.matrix(x).T
        ks = train_x@x
        # Predictive value.
        wx = np.matrix(alphas*train_y)*ks
        fx = wx + b
        return fx[0, 0]

def smo(train_x, train_y, c, iter_num):
    n_train, n_feature = train_x.shape
    alphas = np.zeros(n_train)
    b = 0
    it = 0
    while it < iter_num:
        pair_changed = 0
        for i in range(n_train):
            a_i, x_i, y_i = alphas[i], train_x[i], train_y[i]
            fx_i = f(x_i, train_x, train_y, alphas, b)
            E_i = fx_i - y_i
            j = select_j(i, n_train)
            a_j, x_j, y_j = alphas[j], train_x[j], train_y[j]
            fx_j = f(x_j, train_x, train_y, alphas, b)
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
            #print(abs(a_j_new - a_j_old))
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
        print('iteration number: {}'.format(it))
    return alphas, b

def get_w(alphas, dataset, labels):
    ''' 通过已知数据点和拉格朗日乘子获得分割超平面参数w
    '''
    alphas, dataset, labels = np.array(alphas), np.array(dataset), np.array(labels)
    yx = (labels.reshape(1, -1).T*np.ones(dataset.shape[1],dtype=int))*dataset
    w = np.dot(yx.T, alphas)
    return w.tolist()


#loading data
print("Loading data...")
fo1 = open("data/data/SST-2/train.tsv","r")
fo2 = open("data/data/SST-2/dev.tsv","r")
comment1 = fo1.readline()
comment2 = fo2.readline()
list1 = fo1.readlines()
list2 = fo2.readlines()

train_labels = np.zeros(len(list1), dtype="int8")
test_y = np.zeros(len(list2), dtype="int8")
for i in range(len(list1)):
    list1[i] = list1[i].rstrip('\n')
    if(list1[i][-1] == '0'):
        train_labels[i] = -1
    if(list1[i][-1] == '1'):
        train_labels[i] = 1
    list1[i] = list1[i].rstrip('0')
    list1[i] = list1[i].rstrip('1')
    list1[i] = list1[i].rstrip('\t')
    list1[i] = list1[i].rstrip(' ')

for i in range(len(list2)):
    list2[i] = list2[i].rstrip('\n')
    if(list2[i][-1] == '0'):
        test_y[i] = -1
    if(list2[i][-1] == '1'):
        test_y[i] = 1
    list2[i] = list2[i].rstrip('0')
    list2[i] = list2[i].rstrip('1')
    list2[i] = list2[i].rstrip('\t')
    list2[i] = list2[i].rstrip(' ')

fo1.close()
fo2.close()

N_train = 400
N_test = 10000
#load dictionary(including train data and test data)
list_all = list1+list2
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(list_all[:N_test+N_train])
#print(vectorizer.get_feature_names())

#load train data / test data
_X=X.toarray().astype("int8")
train_X = _X[0:N_train,:]
test_X = _X[N_train:,:]
train_y = train_labels[0:N_train]
test_y  = train_labels[N_train:N_train+N_test]
print("Complete!")

index1 = np.where(train_y==1)[0]
doc_freq1 = _X[index1].sum(axis=0)
doc_freq1 = doc_freq1 / _X[index1].shape[0]
index2 = np.where(train_y==-1)[0]
doc_freq2 = _X[index2].sum(axis=0)
doc_freq2 = doc_freq2 / _X[index2].shape[0]
delete = (2==((doc_freq2/2 < doc_freq1).astype("int8") + (doc_freq1 < doc_freq2 * 2).astype("int8")))
reserve_id = np.where(delete == 0)[0]

train_X = train_X[:,reserve_id]
test_X = test_X[:,reserve_id]


time_build_start = time.time()

alphas, b = smo(train_X, train_y, 0.6, 20)
w = np.multiply(alphas, train_y) @ train_X

time_build_end = time.time()

np.save("trained_data/SST_SVM_w.npy", w)
np.save("trained_data/SST_SVM_alphas.npy", alphas)
np.save("trained_data/SST_SVM_b.npy", b)

w=np.load("trained_data/SST_SVM_w.npy")
alphas=np.load("trained_data/SST_SVM_alphas.npy")
b=np.load("trained_data/SST_SVM_b.npy")

# predicting
print("Start predicting...")
predict_y = np.zeros(N_test)
for i in range(N_test):
    if w @ test_X[i].T + b > 0:
        predict_y[i] = 1
    else:
        predict_y[i] = -1
accuracy = (predict_y == test_y).mean()
print("Complete!")
print("accuracy: "+str(accuracy))


time_end=time.time()

print("Building model time: "+str(time_build_end-time_build_start)+" seconds")
print("Predicting time: "+str(time_end-time_build_end)+" seconds")
print('Total time:',time_end-time_start, "seconds")