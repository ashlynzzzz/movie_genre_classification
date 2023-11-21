import os
import numpy as np


def feature_merge(dir: str):
    arr_list = []
    for file in os.listdir(dir):
        arr_list.append(np.load(os.path.join(dir, file)))
    arr_comb = np.concatenate(arr_list)
    
    with open(dir + '.npy', 'wb') as features_file:
        np.save(features_file, arr_comb)


class FeatureConcatenation:
    def __init__(self, text_feature, image_feature):
        self.text_feature = text_feature
        self.image_feature = image_feature

    def concatenate(self):
        return np.concatenate((self.text_feature, self.image_feature), axis=1)
