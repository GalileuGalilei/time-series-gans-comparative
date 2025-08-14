from .lib import TimeGAN
from .options import Options
from core_interfaces import IGenerator
from data.DataLoader import DAPT2020
import numpy as np

class SyntheticGenerator(IGenerator):
    def __init__(self, model_path, data):

        opt = Options().parse()

        opt.seq_len = 30
        opt.data = "dapt2020"
        opt.resume = model_path
        opt.iteration = 500
        opt.hidden_dim = 150
        opt.num_layer = 3  
        opt.module = "gru" 
        opt.batch_size = 128
        opt.z_dim = 10
        opt.label_embed_dim = 10
        opt.num_classes = 5 

        X_data = data.X_train

        self.model = TimeGAN(opt, X_data, data.Y_train)

    def generate(self, labels: list[int]):
        #todo: adicionar label embeding
        data = self.model.generation(labels)
        #data = np.transpose(data, (0, 2, 1))
        #data = np.expand_dims(data, axis=2)

        return data
    
    @property
    def get_name(self):
        return "TimeGAN"
    

if __name__ == "__main__":
    features_to_train = ['Bwd Packets/s', 'Flow Packets/s', 'Src Port', 'Protocol', 'FIN Flag Count', 'SYN Flag Count', 'Timestamp']
    label_column = 'Stage'
    seq_len = 30 
    filename = "data/output.csv"
    
    data_set = DAPT2020(filename, features_to_train, label_column, seq_len, is_train=True, shuffle=True, expand=False)

    generator = SyntheticGenerator("output/TimeGAN/stock/train/weights", data_set.X_train_set)
    synth_data = generator.generate(data_set.Y_test_set)
    print(len(synth_data))