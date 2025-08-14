from TTSCGAN.TransCGAN_model import *
import pandas as pd
from data.DataLoader import *
import numpy as np
from core_interfaces import IGenerator

class SyntheticGenerator(IGenerator):
    def __init__(self, seq_len, num_channels, num_classes, model_path):
        self.seq_len = seq_len
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.model_path = model_path
        self.gen_net = load_model_generator(seq_len, num_channels, num_classes, model_path)

    def generate(self, fake_labels):
        return generate_fake_samples(self.gen_net, fake_labels)

    @property
    def get_name(self):
        return "TTS-CGAN"

def load_model_generator(seq_len, num_channels, num_classes, model_path):
    # Load the model
    gen_net = Generator(seq_len=seq_len, channels=num_channels, num_classes=num_classes, latent_dim=100, data_embed_dim=64,
                    label_embed_dim=32, depth=3, num_heads=2, 
                    forward_drop_rate=0.0, attn_drop_rate=0.0)
    
    
    CGAN_ckp = torch.load(model_path)
    gen_net.load_state_dict(CGAN_ckp['gen_state_dict'])
    #gen_net.eval()

    return gen_net

def generate_fake_samples(gen_net, fake_labels):
    new_samples = len(fake_labels)
    fake_labels = torch.tensor(fake_labels, dtype=torch.long)

    fake_noise = torch.FloatTensor(np.random.normal(0, 1, (new_samples, 100)))
    fake_sigs = gen_net(fake_noise, fake_labels).to('cpu').detach().numpy()
    fake_sigs = fake_sigs.squeeze(axis=2)  # Remove the singleton dimension
    fake_sigs = fake_sigs.transpose(0, 2, 1)  # Transpose to (batch, seq_len, channels)

    return fake_sigs

def save_fake_samples(fake_sigs, fake_labels, output_path, columns=None):
    # Save the data
    if columns != None:
        fake_data = pd.DataFrame(fake_sigs, columns=columns)
    else:
        fake_data = pd.DataFrame(fake_sigs)

    fake_data['label'] = fake_labels
    fake_data.to_csv(output_path, index=False)
