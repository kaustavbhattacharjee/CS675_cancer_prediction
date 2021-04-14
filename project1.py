from sklearn import svm
from array import array
import sys,math,random
#from feature_selection import FeatureSelection

def cross_validation(data_new, label_new, data_val, labels_val):
    classifier = svm.LinearSVC()
    classifier.fit(data_new, label_new)
    prediction = classifier.predict(data_val)
    err = 0
    for i in range(0, len(prediction), 1):
        if (prediction[i] != labels_val[i]):
            err = err + 1
    err = err / len(labels_val)
    #print('Accuracy', (1 - err))

### Start of feature selection
class FeatureSelection():
    def __init__(self, data, label):
        self.my_dataset = data
        self.my_label = label
        self.row = len(self.my_dataset)
        self.col = len(self.my_dataset[0])

    def mean(self, my_list):
        mean = 0
        for i in my_list:
            mean += i
        return sum(my_list) / len(my_list)
        # return mean / len(my_list)

    def f_scoreCal(self, col):
        col_list = [abcd[col] for abcd in self.my_dataset]
        x_mean = self.mean(col_list)
        x0 = []
        x1 = []
        n0 = 0
        for i in range(self.row):
            if self.my_label[i] == 0:
                x0.append(col_list[i])
                n0 += 1
            else:
                x1.append(col_list[i])
        x0_mean = self.mean(x0)
        x1_mean = self.mean(x1)

        v0 = 0
        v1 = 0
        v0 = sum([(i - x0_mean) ** 2 for i in x0])
        v1 = sum([(i - x1_mean) ** 2 for i in x1])
        '''
        for i in x0:
            v0 += (i - x0_mean) ** 2
        for i in x1:
            v1 += (i - x1_mean) ** 2
        '''
        v0 /= (n0 - 1)
        v1 /= ((self.row - n0) - 1)
        if (v0 + v1) == 0:
            return -1
        return (((x0_mean - x_mean) ** 2 + (x1_mean - x_mean) ** 2) / (v0 + v1))  # f-score formula

    def column_selection(self):

        f_score = {}
        for j in range(self.col):
            f = self.f_scoreCal(j)
            f_score[f] = j
        vals = sorted(f_score.keys(), reverse=True)[:15]  # selecting top 15 f-scores
        featured_col = []
        for i in vals:
            featured_col.append(f_score[i])
        print(' '.join(map(str, featured_col)))
        # print('Below are the feature selected as per f-score\n{}'.format(featured_col))
        return featured_col
### End of feature selection

# Reading training data
#print("The traindata file is being read ...")
my_dataset = []
with open(sys.argv[1], 'r') as file:
    for line in file:
        try:
            my_dataset.append([float(item.strip()) for item in line.strip().split()])
        except:
            print(my_dataset)
            continue

file.close()

# Reading the Labels
#print("The trueclass file is being read ...")
my_label = {}
with open(sys.argv[2]) as f:
    x = f.readline()
    while x != '':
        a = x.split()
        my_label[int(a[1])] = int(a[0])
        x = f.readline()


random.seed()
rowIDs = []
for i in range(0, len(my_dataset), 1):
    rowIDs.append(i)  # creating an array containing rowIDs for all data in the training dataset
    
#### Random 90:10 split for train and validation
training_new = []
labels_new = []
validation = []
labels_validation = []

random.shuffle(rowIDs)  # re-ordering the rowIDs using random shuffle, though random is actually of no use here


for i in range(0, int(.9 * len(rowIDs)), 1):
    training_new.append(my_dataset[i])
    labels_new.append(my_label[i])
for i in range(int(.9 * len(rowIDs)), len(rowIDs), 1):
    validation.append(my_dataset[i])
    labels_validation.append(my_label[i])

#print('Feature Selection part starts ...')
Obj = FeatureSelection(training_new, labels_new)
featured_columns = Obj.column_selection()
# print(featured_columns)
feature_selected_data = []
for my_list in training_new:
    l = []
    for i in featured_columns:
        l.append(my_list[i])
    feature_selected_data.append(l)
data_val = []
for my_list in validation:
    l = []
    for i in featured_columns:
        l.append(my_list[i])
    data_val.append(l)

cross_validation(feature_selected_data, labels_new, data_val, labels_validation)
#print('Reading testdata file...')
testdata = []
with open(sys.argv[3], 'r') as file:
    for line in file:
        testdata.append([float(item.strip()) for item in line.strip().split()])
file.close()
test = []
for my_list in testdata:
    l = []
    for i in featured_columns:
        l.append(my_list[i])
    test.append(l)
classifier = svm.LinearSVC(C=0.01)
classifier.fit(feature_selected_data, labels_new)
prediction = classifier.predict(test)
for i in range(len(prediction)):
    print(str(prediction[i]) + ' ' + str(i))
exit(0)
