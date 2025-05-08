from tts_cgan.TransCGAN_model import *
import pandas as pd
from data.DataLoader import *
import numpy as np

def load_model_generator(seq_len, num_channels, num_classes, model_path):
    # Load the model
    gen_net = Generator(seq_len=seq_len, channels=num_channels, num_classes=num_classes, latent_dim=100, data_embed_dim=64,
                    label_embed_dim=16 ,depth=3, num_heads=8, 
                    forward_drop_rate=0.1, attn_drop_rate=0.1)
    
    CGAN_ckp = torch.load(model_path)
    gen_net.load_state_dict(CGAN_ckp['gen_state_dict'])

    return gen_net

def generate_fake_samples(gen_net, fake_labels):
    #set seed
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    new_samples = len(fake_labels)
    fake_labels = torch.tensor(fake_labels, dtype=torch.long)

    fake_noise = torch.FloatTensor(np.random.normal(0, 1, (new_samples, 100)))
    fake_sigs = gen_net(fake_noise, fake_labels).to('cpu').detach().numpy()

    return fake_sigs

def save_fake_samples(fake_sigs, fake_labels, output_path, columns=None):
    # Save the data
    if columns != None:
        fake_data = pd.DataFrame(fake_sigs, columns=columns)
    else:
        fake_data = pd.DataFrame(fake_sigs)

    fake_data['label'] = fake_labels
    fake_data.to_csv(output_path, index=False)

"""
Recreate the dataset using the generator model following the order of the original dataset labels
"""
def recreate_dataset(data_path, model_path, features_names, seq_len, attack_only=False, shuffle=True):
    train_set = load_and_preprocess_data(data_path, features_names, "Stage", seq_len, attack_only=attack_only, is_train=True, shuffle=shuffle) 

    y_train = train_set.Y_set
    x_train = train_set.X_set

    num_classes = max(y_train) + 1
    num_channels = x_train.shape[1]

    gen_net = load_model_generator(seq_len=seq_len, num_channels=num_channels, num_classes=num_classes, model_path=model_path)
    fake_sigs = generate_fake_samples(gen_net, y_train)

    train_set = SynthDataset(fake_sigs, y_train)

    return train_set

def recreate_balanced_dataset(train_set, gen_net, target_ratios):
    """
    target_ratios: dicionário com {classe: proporção desejada no total final}
    """
    y_train = train_set.Y_set
    x_train = train_set.X_set

    num_classes = max(y_train) + 1
    num_channels = x_train.shape[1]
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
        return train_set

    # Agora calcula quantas amostras gerar para cada classe proporcionalmente
    y_synthetic = []
    for cls, target_ratio in target_ratios.items():
        c_i = current_counts[cls]
        t_i = target_ratio
        extra_needed = (t_i * total_original - c_i) / (1 - t_i)
        if extra_needed > 0:
            y_synthetic.extend([cls] * int(np.ceil(extra_needed)))

    y_synthetic = np.array(y_synthetic)
    fake_sigs = generate_fake_samples(gen_net, y_synthetic)

    print(f"Total de amostras sintéticas geradas: {len(fake_sigs)}")

    x_train = np.concatenate((x_train, fake_sigs), axis=0)
    y_train = np.concatenate((y_train, y_synthetic), axis=0)

    return SynthDataset(x_train, y_train)


def create_mixed_dataset(data_path, model_path, features_names, seq_len, increase_multiplier, atack_only=False):
    train_set = load_and_preprocess_data(data_path, list(features_names), "Stage", seq_len, is_train=True, attack_only=atack_only, shuffle=True) 

    y_train = train_set.Y_set
    x_train = train_set.X_set 

    num_classes = 5
    num_channels = x_train.shape[1]

    #increase the number of labels proportionally to the increase_multiplier. increase_multiplier can be a decimal number
    original_len = len(y_train)
    extra_len = int(original_len * increase_multiplier)
    #new_len = original_len - extra_len
    y_train_fake = np.random.choice(y_train, size=extra_len, replace=True)

    gen_net = load_model_generator(seq_len=seq_len, num_channels=num_channels, num_classes=num_classes, model_path=model_path)
    fake_sigs = generate_fake_samples(gen_net, y_train_fake)

    if atack_only: #como o dataset gerado nao tem benigno, carregamos de novo normalmente
        train_set = load_and_preprocess_data(data_path, list(features_names), "Stage", seq_len, is_train=True, attack_only=False, shuffle=True) 
        x_train = train_set.X_set
        y_train = train_set.Y_set

    #concatenate the original dataset with the new one
    x_train = np.concatenate((x_train, fake_sigs), axis=0)
    y_train = np.concatenate((y_train, y_train_fake), axis=0)

    train_set = SynthDataset(x_train, y_train)

    return train_set

class SynthDataset(Dataset):
    def __init__(self, X_train, Y_train):
        self.X_set = X_train
        self.Y_set = Y_train

    def __len__(self):
        return len(self.X_set)

    def __getitem__(self, idx):
        return self.X_set[idx], self.Y_set[idx]

def main():
    fake_sigs, labels = recreate_dataset()
    save_fake_samples(fake_sigs, labels, "data/fake_data.csv")

if __name__ == "__main__":
    main()