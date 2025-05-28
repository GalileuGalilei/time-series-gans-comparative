from .data_utils import generate_synthetic
from core_interfaces import IGenerator
import numpy as np

class SyntGenerator(IGenerator):
    def __init__(self, model_path, epoch):
        self.epoch = epoch
        self.model_path = model_path

    def generate(self, fake_labels):
        label_size = max(fake_labels) + 1
        #makes the one-hot encoding of the labels
        fake_labels = [[1 if i == label else 0 for i in range(label_size)] for label in fake_labels]
        #generates the synthetic data
        data = generate_synthetic(fake_labels, self.model_path, self.epoch)
        #for compatibility with the other generators, add a new dimention -> (lenght, sequence_length, 1, num_channels)
        data = np.transpose(data, (0, 2, 1))
        data = np.expand_dims(data, axis=2)
        return data

    @property
    def get_name(self):
        return "RCGAN Generator"
