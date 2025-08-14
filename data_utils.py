import TTSCGAN.generate_data as TTSCGAN
import RGAN as RCGAN
import TimeGAN
from data.DataLoader import *
from core_interfaces import IGenerator
from torch.utils.data import Dataset, DataLoader

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

def load_original_dataset(is_train, attack_only=False, shuffle=True):
    features_to_train = ['Src Port', 'Dst Port', 'Bwd Init Win Bytes', 'Flow Packets/s', 'Fwd Packets/s', 'Bwd Packets/s', 'Flow IAT Mean', 'Bwd Header Length', 'Fwd Header Length', 'Flow Bytes/s']
    label_column = 'Stage'
    filename = "data/dapt2020.csv"
    seq_len = 30
    
    data_set = DAPT2020(filename, label_column, seq_len, filter_features=features_to_train, is_train=is_train, attack_only=attack_only)
    if shuffle:
        data_set.shuffle()

    return DataLoader(data_set, batch_size=64)

def generate_semi_syntetic_dataset(original_dataset : DAPT2020, generator : IGenerator, target_ratios):
    """
    target_ratios: dicionário com {classe: proporção desejada no total final}
    """
    y_train = original_dataset.Y_train
    x_train = original_dataset.X_train

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

def generate_syntetic_dataset(orignal_dataset : DAPT2020, generator : IGenerator, shuffle=True):
    """
    Recreate the dataset using the generator model following the order of the original dataset labels
    """
    y_train = orignal_dataset.Y_train

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

''' código de produção, deveria ser removido depois de testar
def fine_tuning(model):
    """
    target_ratios: dicionário com {classe: proporção desejada no total final}
    """
    grow_steps = [0.1, 0.15, 0.25, 0.3, 0.4]
    target_ratios = []
    i = 0
    while i < len(grow_steps)**4:
        target_ratios.append(
            {0: 0, 1: grow_steps[i%len(grow_steps)],
             2: grow_steps[(i//len(grow_steps))%len(grow_steps)],
             3: grow_steps[(i//(len(grow_steps)**2))%len(grow_steps)],
             4: grow_steps[(i//(len(grow_steps)**3))%len(grow_steps)]})
        i += 1

    print(f"Testando um total de target_ratios: {len(target_ratios)}")


    model_path = "TTSCGAN/logs/TTS_APT_CGAN_OITO_VAR_IMPR7/Model/checkpoint"
    trainable_features = ['SYN Flag Count', 'Src Port', 'Fwd Packets/s', 'Flow Packets/s', 'Bwd Packets/s', 'ACK Flag Count', 'FIN Flag Count', 'Flow Bytes/s', 'Timestamp']
    data_path = "data/output.csv"
    seq_len = 30
    scores = []

    best_target_ratio = {}
    best_score = 0

    test_set = load_original_dataset(is_train=False, attack_only=False)
    train_set = load_and_preprocess_data(data_path, list(trainable_features), "Stage", seq_len, is_train=True)
    generator = TTSCGANGenerator(seq_len=seq_len, num_channels=len(trainable_features)-1, num_classes=5, model_path=model_path)

    for target_ratio in target_ratios:
        print(f"Target ratio: {target_ratio}")
        train_set_increased = generate_semi_syntetic_dataset(train_set, generator, target_ratio)

        # roda o modelo
        trained_model = train_cpu_model(train_set_increased.dataset.X_train_set, train_set_increased.dataset.Y_train_set, model.copy())
        #retorna f1 score
        accuracy_score, precision, recall, f1 = evaluate_cpu_model(test_set.dataset.X_test_set, test_set.dataset.Y_test_set, trained_model)
        scores.append(f1)

        if f1 > best_score:
            best_score = f1
            best_target_ratio = target_ratio
    print(f"Melhor target ratio: {best_target_ratio} com f1 score: {best_score}")
'''