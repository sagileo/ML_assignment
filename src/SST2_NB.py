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
#load dictionary, including train data and test data
list_all = list1+list2
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(list_all)

#load train data / test data
train_X = X.toarray()[0:N_train,:]
test_X = X.toarray()[N_train:,:]

_X=X.toarray().astype("int8")
index1 = np.where(train_y==1)[0]
doc_freq1 = _X[index1].sum(axis=0)
doc_freq1 = doc_freq1 / _X[index1].shape[0]
index2 = np.where(train_y==0)[0]
doc_freq2 = _X[index2].sum(axis=0)
doc_freq2 = doc_freq2 / _X[index2].shape[0]
delete = (2==((doc_freq2/2 < doc_freq1).astype("int8") + (doc_freq1 < doc_freq2 * 2).astype("int8")))
reserve_id = np.where(delete == 0)[0]

train_X = train_X[:,reserve_id]
test_X = test_X[:,reserve_id]

print("Complete!")


#calculate posterior probabilities
N_dict = train_X.shape[1]
prob = np.zeros((2, N_dict))

for i in range(2):
    prob[i] = ((train_y == i) @ train_X + 1) / (train_y == i).sum()

log_prob = np.log(prob)
print("Complete!")

#predict
print("Start predicting...")
predict = np.zeros((2, N_test))
predict_y = np.zeros(N_test)
for i in range(2):
    predict[i] = np.multiply(test_X, log_prob[i]).sum(axis = 1).transpose()
    
for i in range(N_test):
    predict_y[i] = np.where(predict[:,i] == np.max(predict[:,i]))[0][0]

accuracy = (predict_y == test_y).mean()
print("Complete!")
print("Accuracy:"+str(accuracy))

time_end=time.time()
print('total time:',time_end-time_start, "seconds")