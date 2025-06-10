from RGAN import RCGAN_generator as RCGANGEN
from TTSCGAN.generate_data import *
from data.DataLoader import *

class SynthDataset(Dataset):
    """
    Synthetic Dataset for training and testing the generator models.

    --- Remarks
    While the syntethic dataset has only (X_train_set, Y_train_set), the original dataset (created by "load_and_preprocess_data") has 
    the (X_train_set, Y_train_set, X_test_set, Y_test_set) attributes and also the (X_set, Y_set) attributes that represents the whole dataset
    """
    def __init__(self, X_train, Y_train):
        self.X_train_set = X_train
        self.Y_train_set = Y_train

    def __len__(self):
        return len(self.X_train_set)

    def __getitem__(self, idx):
        return self.X_train_set[idx], self.Y_train_set[idx]

def load_original_dataset(is_train, attack_only=False, Shuffle=True, expand=True):
    features = ['SYN Flag Count', 'Src Port', 'Fwd Packets/s', 'Flow Packets/s', 'Bwd Packets/s', 'ACK Flag Count', 'FIN Flag Count', 'Flow Bytes/s', 'Timestamp']
    label_column = 'Stage'
    seq_len = 30
    filename = "data/output.csv"
    
    #ja embaralhado por padrao
    data_set = load_and_preprocess_data(filename, features, label_column, seq_len, is_train=is_train, attack_only=attack_only, shuffle=Shuffle, expand=expand)

    return DataLoader(data_set, batch_size=64)

def generate_semi_syntetic_dataset(original_dataset, generator : IGenerator, target_ratios):
    """
    target_ratios: dicionário com {classe: proporção desejada no total final}
    """
    y_train = original_dataset.Y_train_set.copy()
    x_train = original_dataset.X_train_set.copy()

    num_classes = max(y_train) + 1
    total_original = len(y_train)
    current_counts = np.bincount(y_train, minlength=num_classes)

    # Calcular quanto precisamos adicionar no total
    total_extra = 0
    for cls, target_ratio in target_ratios.items():
        c_i = current_counts[cls]
        t_i = target_ratio
        extra_needed = (t_i * total_original - c_i) / (1 - t_i)
        if extra_needed > 0:
            total_extra += int(np.ceil(extra_needed))

    if total_extra == 0:
        print("Nenhuma amostra sintética necessária.")
        return original_dataset

    # Agora calcula quantas amostras gerar para cada classe proporcionalmente
    y_synthetic = []
    for cls, target_ratio in target_ratios.items():
        c_i = current_counts[cls]
        t_i = target_ratio
        extra_needed = (t_i * total_original - c_i) / (1 - t_i)
        if extra_needed > 0:
            y_synthetic.extend([cls] * int(np.ceil(extra_needed)))

    y_synthetic = np.array(y_synthetic)
    fake_sigs = generator.generate(y_synthetic)

    print("Generating SEMI synthetic dataset")

    x_train = np.concatenate((x_train, fake_sigs), axis=0)
    y_train = np.concatenate((y_train, y_synthetic), axis=0)

    print(f"Total synthetic samples generated: {len(y_synthetic)}")

    # Embaralhar os dados
    np.random.seed(22)
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]

    return DataLoader(SynthDataset(x_train, y_train), batch_size=64)

def generate_syntetic_dataset(orignal_dataset, generator : IGenerator, shuffle=True):
    """
    Recreate the dataset using the generator model following the order of the original dataset labels
    """
    y_train = orignal_dataset.Y_train_set.copy()

    fake_sigs = generator.generate(y_train)
    syntetic_dataset = SynthDataset(fake_sigs, y_train)

    if shuffle:
        # Embaralhar os dados
        np.random.seed(22)
        indices = np.arange(len(syntetic_dataset.X_train_set))
        np.random.shuffle(indices)
        syntetic_dataset.X_train_set = syntetic_dataset.X_train_set[indices]
        syntetic_dataset.Y_train_set = syntetic_dataset.Y_train_set[indices]

    print("Generating synthetic dataset")
    return DataLoader(syntetic_dataset, batch_size=64)