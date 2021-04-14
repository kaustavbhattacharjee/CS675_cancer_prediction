
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
        return sum(my_list)/len(my_list)
        #return mean / len(my_list)


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
        return (((x0_mean - x_mean) ** 2 + (x1_mean - x_mean) ** 2) / (v0 + v1)) #f-score formula

    def column_selection(self):

        f_score = {}
        for j in range(self.col):
            f = self.f_scoreCal(j)
            f_score[f] = j
        vals = sorted(f_score.keys(), reverse=True)[:15] # selecting top 15 f-scores
        featured_col = []
        for i in vals:
            featured_col.append(f_score[i])
        print(' '.join(map(str,featured_col)))
        #print('Below are the feature selected as per f-score\n{}'.format(featured_col))
        return featured_col