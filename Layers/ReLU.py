import numpy as np
from Layers import Base

class ReLU():

    def __init__(self):
        self.input_size = 5
        self.batch_size = 10
        self.half_batch_size = int(self.batch_size / 2)
        self.input_tensor = np.ones([self.batch_size, self.input_size])
        self.input_tensor[0:self.half_batch_size, :] -= 2



        Trainable = Base.BaseLayer()
        self.trainable = Trainable.trainable


    def forward(self, input_tensor):

        Input_tensor = np.zeros([self.batch_size, self.input_size])

        for i in range(0, self.input_size):
            for j in range(0, self.batch_size):
                if input_tensor[j, i] > 0:
                    Input_tensor[j, i] = input_tensor[j, i]

        return Input_tensor

    def backward(self, error_tensor):

        Error_tensor = np.zeros([self.batch_size, self.input_size])

        for i in range(0, self.input_size):
            for j in range(0, self.batch_size):
                if error_tensor[j, i] > 0:
                    Error_tensor[j, i] = error_tensor[j, i]

        return Error_tensor










