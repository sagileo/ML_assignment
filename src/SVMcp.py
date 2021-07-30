import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import time

time_start = time.time()

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


def kernalTransfrom(dataMatrix, vector, kTup):
    if kTup[0] == "lin":
        return vector * dataMatrix.transpose()
    elif kTup[0] == "rbf":
        delta = dataMatrix - vector
        K = np.matrix(np.diag(delta * delta.transpose()), dtype=np.float)
        K = np.exp(K / (-2 * kTup[1] ** 2))
        return K
    else:
        raise NameError("Kernal Name Error")

class osStruct:
    def __init__(self, dataMatIn, classlabels, C , toler, kTup):
        self.dataMatrix = np.matrix(dataMatIn, dtype=np.float)
        self.labelMatrix = np.matrix(classlabels, dtype=np.float).transpose()
        self.C = C
        self.toler = toler
        self.m = self.dataMatrix.shape[0]
        self.b = 0
        self.alphas = np.matrix(np.zeros((self.m, 1)), dtype=np.float)
        self.eCache = np.matrix(np.zeros((self.m, 2)), dtype=np.float)
        self.K = np.matrix(np.zeros((self.m, self.m)), dtype=np.float)
        for i in range(self.m):
            self.K[i] = kernalTransfrom(self.dataMatrix, self.dataMatrix[i, :], kTup)

def selectJRand(i, m):
    j = i
    while j == i:
        j = np.random.randint(0, m, 1)[0]
    return j

def clipAlpha(alpha, L, H):
    if alpha >= H:
        return H
    elif alpha <= L:
        return L
    else:
        return alpha

def calEi(obj, i):
    fxi = float(np.multiply(obj.alphas, obj.labelMatrix).transpose() * \
                obj.K[:, i]) + obj.b
    Ek = fxi - obj.labelMatrix[i, 0]
    return float(Ek)

def updateEi(obj, i):
    Ei = calEi(obj, i)
    obj.eCache[i] = [1, Ei]

def selectJIndex(obj, i, Ei):
    maxJ = -1
    maxdelta = -1
    Ek = -1
    obj.eCache[i] = [1, Ei]
    vaildEiList = np.nonzero(obj.eCache[:, 0].A)[0]
    if len(vaildEiList) > 1:
        for j in vaildEiList:
            if j == i:
                continue
            Ej = calEi(obj, j)
            delta = np.abs(Ei - Ej)
            if delta > maxdelta:
                maxdelta = delta
                maxJ = j
                Ek = Ej
    else:
        maxJ = selectJRand(i, obj.m)
        Ek = calEi(obj, maxJ)
    return Ek, maxJ

def innerLoop(obj, i):
    Ei = calEi(obj, i)
    if (obj.labelMatrix[i, 0] * Ei < -obj.toler and obj.alphas[i, 0] < obj.C) or \
            (obj.labelMatrix[i, 0] * Ei > obj.toler and obj.alphas[i, 0] > 0):
        Ej, j = selectJIndex(obj, i, Ei)
        alphaIold = obj.alphas[i, 0].copy()
        alphaJold = obj.alphas[j, 0].copy()
        if obj.labelMatrix[i, 0] == obj.labelMatrix[j, 0]:
            L = max(0, obj.alphas[i, 0] + obj.alphas[j, 0] - obj.C)
            H = min(obj.C , obj.alphas[i, 0] + obj.alphas[j, 0])
        else:
            L = max(0, obj.alphas[j, 0] - obj.alphas[i, 0])
            H = min(obj.C, obj.C - obj.alphas[i, 0] + obj.alphas[j, 0])
        if L == H:
            return 0
        eta = obj.K[i, i] + obj.K[j, j] - 2 * obj.K[i, j]
        if eta <= 0:
            return 0
        obj.alphas[j, 0] += obj.labelMatrix[j, 0] * (Ei - Ej) / eta
        obj.alphas[j, 0] = clipAlpha(obj.alphas[j, 0], L, H)
        updateEi(obj, j)
        if np.abs(obj.alphas[j, 0] - alphaJold) < 0.00001:
            return 0
        obj.alphas[i, 0] += obj.labelMatrix[i, 0] * obj.labelMatrix[j, 0] * (alphaJold - obj.alphas[j, 0])
        updateEi(obj, i)
        b1 = -Ei - obj.labelMatrix[i, 0] * obj.K[i, i] * (obj.alphas[i, 0] - alphaIold) \
             - obj.labelMatrix[j, 0] * obj.K[i, j] * (obj.alphas[j, 0] - alphaJold) + obj.b
        b2 = -Ej - obj.labelMatrix[i, 0] * obj.K[i, j] * (obj.alphas[i, 0] - alphaIold) \
             - obj.labelMatrix[j, 0] * obj.K[j, j] * (obj.alphas[j, 0] - alphaJold) + obj.b
        if obj.alphas[i, 0] > 0 and obj.alphas[i, 0] < obj.C:
            obj.b = b1
        elif obj.alphas[j, 0] > 0 and obj.alphas[j, 0] < obj.C:
            obj.b = b2
        else:
            obj.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

def realSMO(trainSet, trainLabels, C, toler, kTup=('lin', 1.3), maxIter=40):
    obj = osStruct(trainSet, trainLabels, C, toler, kTup)
    entrySet = True
    iterNum = 0
    alphapairschanged = 0
    while (iterNum < maxIter) and (alphapairschanged > 0 or entrySet):
        print(iterNum)
        alphapairschanged = 0
        if entrySet:
            for i in range(obj.m):
                alphapairschanged += innerLoop(obj, i)
                if i % 100 == 0:
                    print("full set loop, iter: %d, alphapairschanged: %d, iterNum: %d" % (i, alphapairschanged, iterNum))
            iterNum += 1
        else:
            vaildalphsaList = np.nonzero((obj.alphas.A > 0) * (obj.alphas.A < C))[0]
            for i in vaildalphsaList:
                alphapairschanged += innerLoop(obj, i)
                if i % 100 == 0:
                    print("non-bound set loop, iter: %d, alphapairschanged: %d, iterNum: %d" % (i, alphapairschanged, iterNum))
            iterNum += 1
        if entrySet:
            entrySet = False
        elif alphapairschanged == 0:
            entrySet = True
            print("iter num: %d" % (iterNum))
    return obj.alphas, obj.b

def getSupportVectorandSupportLabel(trainSet, trainLabel, alphas):
    vaildalphaList = np.nonzero(alphas)[0]
    dataMatrix = np.matrix(trainSet, dtype=np.float)
    labelMatrix = np.matrix(trainLabel, dtype=np.float).transpose()
    sv = dataMatrix[vaildalphaList]#得到支持向量
    svl = labelMatrix[vaildalphaList]
    return sv, svl

def predictLabel(data, sv, svl, alphas, b, kTup):
    kernal = kernalTransfrom(sv, np.matrix(data, dtype=np.float), kTup).transpose()
    fxi = np.multiply(svl.T, alphas[alphas != 0]) * kernal + b
    return np.sign(fxi)


if '__main__' == __name__:
    #loading data
    print("Loading data...")
    fo1 = open("data/data/SST-2/train.tsv","r")
    fo2 = open("data/data/SST-2/dev.tsv","r")
    comment1 = fo1.readline()
    comment2 = fo2.readline()
    list1 = fo1.readlines()
    list2 = fo2.readlines()

    train_y = np.zeros(len(list1), dtype="int8")
    test_y = np.zeros(len(list2), dtype="int8")
    for i in range(len(list1)):
        list1[i] = list1[i].rstrip('\n')
        if(list1[i][-1] == '0'):
            train_y[i] = -1
        if(list1[i][-1] == '1'):
            train_y[i] = 1
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

    N_train = len(list1)
    N_test = len(list2)
    #load dictionary(including train data and test data)
    list_all = list1+list2
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(list_all)
    #print(vectorizer.get_feature_names())

    #load train data / test data
    _X=X.toarray().astype("int8")
    train_X = _X[0:N_train,:]
    test_X = _X[N_train:,:]
    print("Complete!")

    # 加载训练数据
    #dataset, labels = load_data('data/data/testSet.txt')
    time_build_start = time.time()
    alphas, b = realSMO(train_X[:5000,:], train_y[:5000], 0.6, 0.0001)
    time_build_end = time.time()
    
    np.save("trained_data/SST_SVM_alphas.npy", alphas)
    np.save("trained_data/SST_SVM_b.npy", b)

    alphas=np.load("trained_data/SST_SVM_alphas.npy")
    b=np.load("trained_data/SST_SVM_b.npy")

    sv, svl = getSupportVectorandSupportLabel(train_X[:5000,:], train_y[:5000], alphas)
    predict_y =  predictLabel(test_X, sv, svl, alphas, b, kTup=('lin', 1.3))
    print("accuracy: "+str((test_y == predict_y).mean()))

    time_end=time.time()

    print("Building model time: "+str(time_build_end-time_build_start)+" seconds")
    print("Predicting time: "+str(time_end-time_build_end)+" seconds")
    print('Total time:',time_end-time_start, "seconds")
