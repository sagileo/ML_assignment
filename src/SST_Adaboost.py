from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import time

time_start=time.time()




#loading data
print("Loading data...")
fo1 = open("data/data/SST-2/train.tsv","r")
fo2 = open("data/data/SST-2/dev.tsv","r")
comment1 = fo1.readline()
comment2 = fo2.readline()
list1 = fo1.readlines()
list2 = fo2.readlines()

train_y = np.zeros(len(list1))
test_y = np.zeros(len(list2))
for i in range(len(list1)):
    list1[i] = list1[i].rstrip('\n')
    if(list1[i][-1] == '0'):
        train_y[i] = 0
    if(list1[i][-1] == '1'):
        train_y[i] = 1
    list1[i] = list1[i].rstrip('0')
    list1[i] = list1[i].rstrip('1')
    list1[i] = list1[i].rstrip('\t')
    list1[i] = list1[i].rstrip(' ')

for i in range(len(list2)):
    list2[i] = list2[i].rstrip('\n')
    if(list2[i][-1] == '0'):
        test_y[i] = 0
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
train_X = X.toarray()[0:N_train,:]
test_X = X.toarray()[N_train:,:]
print("Complete!")

#calculate frequencies of each label
print("Calculating parameters...")
freq = np.zeros(2)
freq[1] = train_y.mean()
freq[0] = 1- train_y.mean()

#calculate posterior probabilities
N_dict = train_X.shape[1]

def make_pred(classifier_id,  prob):
    predict = np.zeros((2, N_test))
    predict_y = np.zeros(N_test)
    for j in range(2):
        predict[j] = np.multiply(test_X, prob[classifier_id, j]).sum(axis = 1).transpose()
    for j in range(N_test):
        predict_y[j] = np.where(predict[:,j] == np.max(predict[:,j]))[0][0]
    return predict_y

iter_num = 200
d = np.ones(N_train) / N_train
train_predict = np.zeros((2, N_train))
train_predict_y = np.zeros(N_train)
prob = np.zeros((iter_num, 2, N_dict))
alpha = np.zeros(iter_num)
predict = np.zeros((0, N_test))
for i in range(iter_num):
    print("iter "+str(i))
    weighted_X = np.multiply(np.mat(d).T,train_X) * N_train
    tj =(np.vstack((train_y,train_y)) == np.mat([0,1]).T)
    prob[i] =(tj @ weighted_X + 1) / tj.sum(axis=1)
    prob[i] = np.log(prob[i])
    for j in range(2):
        train_predict[j] = np.multiply(train_X, prob[i,j]).sum(axis = 1).transpose()
    for j in range(N_train):
        train_predict_y[j] = np.where(train_predict[:,j] == np.max(train_predict[:,j]))[0][0]
    error_rate = 1 - d[np.where(train_predict_y == train_y)[0]].sum()
    print("train error rate: "+str(error_rate))
    alpha[i] = 0.5 * np.log((1-error_rate)/error_rate)
    d_ = np.multiply(d, np.exp(-alpha[i] * ((train_y == train_predict_y).astype(int) - (train_y != train_predict_y).astype(int))))
    d = d_ / d_.sum()
    predict_y = np.zeros(N_test)
    s = np.zeros(2)
    predict_i = make_pred(i, prob)
    predict = np.vstack((predict, predict_i))
    for j in range(N_test):
        maxsum = -np.inf
        maxk = 0
        for k in range(2):
            s = np.multiply((predict[:,j] == k).astype(float), alpha[:i+1]).sum()
            if maxsum < s:
                maxsum = s
                maxk = k
        predict_y[j] = maxk
    error_rate = 1 - (predict_y == test_y).mean()
    print("test error rate: "+str(error_rate))
    fo = open("results/SST_Adaboost_test_error_rate.txt", "a")
    fo.write(str(i)+"\t"+str(error_rate)+"\n")

print("Complete!")

#predict
print("Start predicting...")

predict = np.zeros((iter_num, N_test))
predict_y = np.zeros(N_test)
s = np.zeros(2)
for i in range(iter_num):
    predict[i] = make_pred(i, prob)
for i in range(N_test):
    for k in range(2):
        s[k] = np.multiply((predict[:,i] == k).astype(float), alpha).sum()
    predict_y[i] = np.where(s == s.max())[0][0]
error_rate = 1 - (predict_y == test_y).mean()
print("test error rate: "+str(error_rate))

accuracy = (predict_y == test_y).mean()
print("Complete!")
print("Accuracy:"+str(accuracy))

time_end=time.time()
print('total time:',time_end-time_start, "seconds")