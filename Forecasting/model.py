import os
import tensorflow as tf
import numpy as np
from keras.utils import to_categorical

class model:
    def __init__(self, path):
        self.model1 = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel_46'))
        self.model2 = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel_49'))

    def predict(self, X, categories):

        categories_dict = {
            'A': 0,
            'B': 1,
            'C': 2,
            'D': 3,
            'E': 4,
            'F': 5
        }
        categories = np.array([categories_dict[letter] for letter in categories])
        categories = to_categorical(categories, num_classes=6)

        alpha = 0.4693877551020408

        # Model 1
        out11 = self.model1.predict([X, categories])  # Shape [BSx9]

        # for each element in out11, if the class is A or E, multiply by 0.8 else multiply by 0.2
        for i in range(len(out11)):
            if categories[i][0] == 1 or categories[i][4] == 1:
                out11[i] *= alpha
            else:
                out11[i] *= 1 - alpha


        # Model 2
        out12 = self.model2.predict([X, categories])  # Shape [BSx9]

        # for each element in out12, if the class is A or E, multiply by 0.2 else multiply by 0.8
        for i in range(len(out12)):
            if categories[i][0] == 1 or categories[i][4] == 1:
                out12[i] *= 1 - alpha
            else:
                out12[i] *= alpha


        # Sum the two outputs
        out1 = out11 + out12
        
        
        X = np.concatenate((X[:,9:],out1), axis = 1)
        
        
        # Model 1
        out21 = self.model1.predict([X, categories])  # Shape [BSx9]

        # for each element in out21, if the class is A or E, multiply by 0.8 else multiply by 0.2
        for i in range(len(out21)):
            if categories[i][0] == 1 or categories[i][4] == 1:
                out21[i] *= alpha
            else:
                out21[i] *= 1 - alpha


        # Model 2
        out22 = self.model2.predict([X, categories])  # Shape [BSx9]

        # for each element in out22, if the class is A or E, multiply by 0.2 else multiply by 0.8
        for i in range(len(out22)):
            if categories[i][0] == 1 or categories[i][4] == 1:
                out22[i] *= 1 - alpha
            else:
                out22[i] *= alpha

        out2 = out21 + out22

        
        
        final_pred = np.concatenate((out1, out2), axis = 1)
     
        
        return final_pred