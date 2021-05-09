import numpy as np
import math
class CrossEntropyLoss():

    def __init__(self):
        self.batch_size = 9
        self.categories = 4
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1


    def forward(self, input_tensor, label_tensor):
        #print(str(label_tensor)+'\n')
        L = np.zeros([self.batch_size, self.categories])
        eps = np.finfo(L.dtype).eps

        #print('L IST'+str(eps) +'\n')

        for i in range(0, self.batch_size):
            for j in range(0 ,self.categories):
                #print(str(label_tensor[i, j]) +'\n')
                if label_tensor[i, j]== 1:

                    L[i, j] = -(math.log(input_tensor[i, j]+ eps))
                    #print('L IS' + str(L[i, j])+ '\n')

        crossentropyloss = np.sum(L,axis=None)
        return crossentropyloss

    def backward(self, error_tensor):
        E = np.zeros([self.batch_size, self.categories])



        return error_tensor