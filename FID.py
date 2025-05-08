from torch.utils.data import TensorDataset, DataLoader
from tts_cgan.generate_data import recreate_dataset
from data.DataLoader import load_and_preprocess_data
import torch
import torch.nn as nn
import numpy as np
from scipy import linalg

class TemporalEncoder(nn.Module):
    def __init__(self, input_channels, seq_length, embedding_dim=128):
        super(TemporalEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, embedding_dim)
        )

    def forward(self, x):
        return self.encoder(x)
    

def calculate_fid(mu1, sigma1, mu2, sigma2):
    # Calcula FID com a fórmula original
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    
    # Corrige imaginação numérica
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


def compute_embeddings(encoder, data_loader, device='cpu'):
    encoder.eval()
    embeddings = []

    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            batch = batch.to(device)
            emb = encoder(batch)
            embeddings.append(emb.cpu().numpy())

    return np.vstack(embeddings)


def calculate_fid_for_gan(real_data, fake_data, batch_size=64, device='cuda'):
    encoder = TemporalEncoder(input_channels=real_data.shape[1], seq_length=real_data.shape[-1])
    encoder = encoder.to(device)

    real_loader = DataLoader(TensorDataset(torch.tensor(real_data).float()), batch_size=batch_size)
    fake_loader = DataLoader(TensorDataset(torch.tensor(fake_data).float()), batch_size=batch_size)

    #real_embs = compute_embeddings(encoder, real_loader, device)
    #fake_embs = compute_embeddings(encoder, fake_loader, device)

    #calculate embeddings by using the channels in the same dimension
    real_embs = compute_embeddings(encoder, real_loader, device).reshape(real_data.shape[0], -1)
    fake_embs = compute_embeddings(encoder, fake_loader, device).reshape(fake_data.shape[0], -1)

    mu_real, sigma_real = real_embs.mean(axis=0), np.cov(real_embs, rowvar=False)
    mu_fake, sigma_fake = fake_embs.mean(axis=0), np.cov(fake_embs, rowvar=False)

    fid = calculate_fid(mu_real, sigma_real, mu_fake, sigma_fake)
    return fid

def main():
    # Load the real data
    data_path = "data/output.csv"
    model_path = "logs/TTS_APT_CGAN_OITO_VAR_IMPR7/Model/checkpoint"
    seq_len = 30
    features_names = ['SYN Flag Count', 'Src Port', 'Fwd Packets/s', 'Flow Packets/s', 'Bwd Packets/s', 'ACK Flag Count', 'FIN Flag Count', 'Flow Bytes/s', 'Timestamp']
    fake_dataset = load_and_preprocess_data(data_path, features_names, "Stage", seq_len, is_train=True)
    features_names = ['SYN Flag Count', 'Src Port', 'Fwd Packets/s', 'Flow Packets/s', 'Bwd Packets/s', 'ACK Flag Count', 'FIN Flag Count', 'Flow Bytes/s', 'Timestamp']
    real_dataset = load_and_preprocess_data(data_path, features_names, "Stage", seq_len, is_train=True)

    # Calculate FID
    fid_score = calculate_fid_for_gan(real_dataset.X_set, fake_dataset.X_set, batch_size=16, device='cuda')
    print(f"FID Score: {fid_score}")

if __name__ == "__main__":
    main()