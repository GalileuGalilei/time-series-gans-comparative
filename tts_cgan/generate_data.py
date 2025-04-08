from tts_cgan.TransCGAN_model import *
import pandas as pd
from data.DataLoader import *
import numpy as np

def load_model_generator(seq_len, num_channels, num_classes, model_path):
    # Load the model
    gen_net = Generator(seq_len=seq_len, channels=num_channels, num_classes=num_classes, latent_dim=100, data_embed_dim=10, 
                    label_embed_dim=10 ,depth=3, num_heads=5, 
                    forward_drop_rate=0.5, attn_drop_rate=0.5)
    
    CGAN_ckp = torch.load(model_path)
    gen_net.load_state_dict(CGAN_ckp['gen_state_dict'])

    return gen_net

def generate_fake_samples(gen_net, fake_labels):
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
def recreate_dataset(data_path, model_path, features_names, seq_len):
    train_set = load_and_preprocess_data(data_path, features_names, "Stage", seq_len, is_train=True) 

    y_train = train_set.Y_set
    x_train = train_set.X_set 

    num_classes = max(y_train) + 1
    num_channels = x_train.shape[1]

    gen_net = load_model_generator(seq_len=seq_len, num_channels=num_channels, num_classes=num_classes, model_path=model_path)
    fake_sigs = generate_fake_samples(gen_net, y_train)

    train_set = SynthDataset(fake_sigs, y_train)

    return train_set


def recreate_increased_dataset(data_path, model_path, features_names, seq_len, increase_multiplier):
    train_set = load_and_preprocess_data(data_path, features_names, "Stage", seq_len, is_train=True) 

    y_train = train_set.Y_set
    x_train = train_set.X_set 

    num_classes = max(y_train) + 1
    num_channels = x_train.shape[1]

    #increase the number of labels proportionally to the increase_multiplier
    y_train = np.repeat(y_train, increase_multiplier)

    gen_net = load_model_generator(seq_len=seq_len, num_channels=num_channels, num_classes=num_classes, model_path=model_path)
    fake_sigs = generate_fake_samples(gen_net, y_train)

    train_set = SynthDataset(fake_sigs, y_train)

    return train_set

def create_mixed_dataset(data_path, model_path, features_names, seq_len, increase_multiplier):
    train_set = load_and_preprocess_data(data_path, features_names, "Stage", seq_len, is_train=True) 

    y_train = train_set.Y_set
    x_train = train_set.X_set 

    num_classes = max(y_train) + 1
    num_channels = x_train.shape[1]

    #increase the number of labels proportionally to the increase_multiplier
    y_train_fake = np.repeat(y_train, increase_multiplier)

    gen_net = load_model_generator(seq_len=seq_len, num_channels=num_channels, num_classes=num_classes, model_path=model_path)
    fake_sigs = generate_fake_samples(gen_net, y_train_fake)

    #concatenate the original dataset with the new one
    x_train = np.concatenate((x_train, fake_sigs), axis=0)
    y_train = np.concatenate((y_train, y_train_fake), axis=0)

    train_set = SynthDataset(x_train, y_train)

    return train_set

class SynthDataset(Dataset):
    def __init__(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.Y_train[idx]

def recreate_balanced_dataset(model_path, features_names, seq_len, n_samples, num_classes, num_channels):
    gen_net = load_model_generator(seq_len=seq_len, num_channels=num_channels, num_classes=num_classes, model_path=model_path)
    
    Y_train = np.array([])
    
    #generate fake labels in Y_train
    for i in range(num_classes):
        Y_train = np.append(Y_train, np.full((n_samples,), i))

    X_train = generate_fake_samples(gen_net, Y_train)

    return SynthDataset(X_train, Y_train)

def main():
    fake_sigs, labels = recreate_dataset()
    save_fake_samples(fake_sigs, labels, "data/fake_data.csv")

if __name__ == "__main__":
    main()