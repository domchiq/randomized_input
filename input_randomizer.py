import numpy as np
from random import randint

class input_randomizer():
    
    """
    class is focused on randomizing input of mnist dataset but might be used for randomizing
    any 3D numpy array with shape (x, 28, 28)
    
    input - x_train: samples of cases to be randomized
          - y_train: labels to samples
          - randomizer_edge: states the max level of arrays cut from the top or from the bottom
            during randomization
    
    """
    
    x_set = np.array([])
    y_set = np.array([])
    
    def __init__(self, x_train, y_train, randomizer_edge=2):
        self.x_train = x_train
        self.y_train = y_train
        self.randomizer_edge = randomizer_edge
     
    def set_init(self, output_length_per_case=1000):
        """
        input - output_length_per_case: number of randomized cases to be created per one case
        method initializes randomized sets
        """
        self.x_set = np.array([case for case in self.x_train for i in range(output_length_per_case)])
        self.y_set = np.array([label for label in self.y_train for i in range(output_length_per_case)])
        
    def image_horizontal_randomizer(self):
        """
        method cuts the arrays either from the top or from the bottom and replaces them with
        arrays filled with zeros
        """
        for no, case in enumerate(self.x_set):
            randomizer = randint(0, self.randomizer_edge)
            if no%2 == 0:
                self.x_set[no] = np.concatenate((case[randomizer:],np.zeros((randomizer, 28))), axis=0)
            else:
                if randomizer != 0:
                    self.x_set[no] = np.concatenate((np.zeros((randomizer, 28)), case[:-randomizer]), axis=0)

    def image_vertical_randomizer(self):
        """
        method cuts the arrays either from the left or from the right and replaces them with
        arrays filled with zeros
        """
        for no, case in enumerate(self.x_set):
            randomizer = randint(0, self.randomizer_edge)
            if no%2 == 0:
                self.x_set[no] = np.concatenate((case[:, randomizer:], np.zeros((28, randomizer))), axis=1)
            else:
                if randomizer != 0:
                    self.x_set[no] = np.concatenate((np.zeros((28, randomizer)), case[:, :-randomizer]), axis=1)

    def image_pixel_shuffling(self, randomizer_edge=50):
        """
        input - randomizer_edge: different to general randomizer_edge, represents top boundary
                                 of the pixel change
        method randomizes value of pixels which should change the edges slightly for every case
        should to be called before set_normalizer
        """

        randomizer = randint(0,randomizer_edge)
        for no, case in enumerate(self.x_set):
            for caseNo, array in enumerate(case):
                for valueNo, value in enumerate(array):
                    if value > 0:
                        if no%2 == 0:
                            if value + randomizer <= 255:
                                self.x_set[no][caseNo][valueNo] = value + randomizer
                            else:
                                self.x_set[no][caseNo][valueNo] = 255
                        else:
                            if value - randomizer > 0:
                                self.x_set[no][caseNo][valueNo] = value - randomizer
                            else:
                                self.x_set[no][caseNo][valueNo] = 0

    def general_randomizer(self, *args):
        """
        takes methods of the class as input and uses all of them on the initialized set
        """
        for function in args:
            function(self)
    
    def set_normalizer(self):
        """
        normalizes the values of pixels
        """
        self.x_set = self.x_set/255
    
    def randomized_set(self):
        """
        returns randomized set with labels
        """
        return self.x_set, self.y_set

