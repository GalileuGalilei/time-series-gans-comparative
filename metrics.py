from data.data_utils import *
from collections import defaultdict
from classifiers.Classifiers import RandomForestClassifierModel, SVMClassifier, LSTMClassifier, TransformerClassifier
from evaluation import *
from TimeGAN import SyntheticGenerator

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from fastdtw import fastdtw as dtw
from scipy.stats import entropy

def plot_samples(data, features_names, labels=None, offset=0, path=None, title="Samples"):
    fig, axs = plt.subplots(2, 2, figsize=(10, 5))
    fig.suptitle(title, fontsize=15)

    # Definição de cores
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    num_samples = data.shape[0]
    num_classes = len(features_names)

    for i in range(2):
        for j in range(2):
            sample_idx = i * 10 + j + offset
            if sample_idx >= num_samples:
                break  # Evita acessar índices fora do alcance de 'data'
            for k in range(1, num_classes):
                if k >= len(colors):
                    break  # Evita acessar índices fora do alcance de 'colors'
                axs[i, j].plot(data[sample_idx, :, k], color=colors[k], label=features_names[k])

            # Alteração da cor de fundo com base no rótulo
            if labels is not None:
                if sample_idx < len(labels):
                    axs[i, j].set_facecolor('white' if labels[sample_idx] == 0 else 'red')

    # Criação da legenda
    handles = [plt.Line2D([0], [0], color=colors[k], lw=2) for k in range(1, num_classes) if k < len(colors)]
    fig.legend(handles, features_names[1:num_classes], loc='upper right', fontsize=12)

    if path:
        if path.endswith('.pdf'):
            plt.savefig(path, format='pdf')
        else:
            plt.savefig(path)

def plot_PCA_TSE(series1, series2, method='both', folder_path='experiments/metrics'):
    """
    Plota a comparação entre duas séries temporais usando PCA e T-SNE.

    Args:
        series1 (np.ndarray): Primeira série temporal (n amostras, d dimensões).
        series2 (np.ndarray): Segunda série temporal (n amostras, d dimensões).
        method (str): 'pca' para apenas PCA, 'tsne' para apenas T-SNE, ou 'both' para ambos.

    Returns:
        None
    """
    assert series1.shape == series2.shape, "As séries devem ter o mesmo formato."

    series1 = series1.reshape(-1, series1.shape[1])
    series2 = series2.reshape(-1, series2.shape[1])

    #pega algumas amostras, aleatoriamente
    n_samples = 500
    indices = np.random.choice(series1.shape[0], n_samples, replace=False)
    series1 = series1[indices]
    series2 = series2[indices]

    # Intercala os dados
    interleaved_data = np.empty((series1.shape[0] + series2.shape[0], series1.shape[1]))
    interleaved_labels = np.empty(series1.shape[0] + series2.shape[0], dtype=int)

    interleaved_data[0::2] = series1
    interleaved_data[1::2] = series2
    interleaved_labels[0::2] = 0  # reais
    interleaved_labels[1::2] = 1  # sintéticos

    # Criando os subplots dinamicamente
    if method == 'both':
        fig, axes = plt.subplots(2, 1, figsize=(6, 12))
        methods = ['PCA', 'T-SNE']
    else:
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        methods = [method.upper()]
        axes = [ax]

    for ax, m in zip(axes, methods):
        if m == 'PCA':
            reducer = PCA(n_components=2)
        elif m == 'T-SNE':
            reducer = TSNE(n_components=2, perplexity=10, random_state=42)
        else:
            raise ValueError("Método inválido. Escolha 'pca', 'tsne' ou 'both'.")

        reduced = reducer.fit_transform(interleaved_data)
        scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=interleaved_labels, cmap='jet', alpha=0.35)
        legend = ax.legend(*scatter.legend_elements())
        ax.add_artist(legend)
        legend.get_texts()[0].set_text("Real")
        legend.get_texts()[1].set_text("Sintético")
        #legenda para os dados artificiais e os reais

        ax.set_title(f"{m}")
    
    #create and save a figure with the two plots
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    # Save the figure as PDF
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
    plt.savefig(folder_path + f"/{current_time}_plot_pca_tse.pdf", format='pdf')
    plt.show()
    return fig, axes

def plot_class_PCA_TSE(series1, series2, labels1, labels2, save_path="images/by_class_pca_tse.pdf", method='both'):
    """
    Plots PCA and/or T-SNE projections of real and synthetic time series, colored by class and domain.

    Args:
        series1 (np.ndarray): Real sequences, shape (n_seq, seq_len, n_feat)
        series2 (np.ndarray): Synthetic sequences, same shape as series1
        labels1 (np.ndarray): Labels for real sequences, shape (n_seq,)
        labels2 (np.ndarray): Labels for synthetic sequences, shape (n_seq,)
        method (str): 'pca', 'tsne', or 'both'
        save_path (str): File path to save the figure

    Returns:
        fig, axes: Matplotlib figure and axes
    """
    assert series1.shape == series2.shape, "series1 and series2 must have the same shape"
    assert labels1.shape[0] == series1.shape[0], "labels1 must match number of sequences in series1"
    assert labels2.shape[0] == series2.shape[0], "labels2 must match number of sequences in series2"

    n_samples = min(500, series1.shape[0])
    indices = np.random.choice(series1.shape[0], n_samples, replace=False)

    # Sample sequences and labels
    series1 = series1[indices]
    series2 = series2[indices]
    labels1 = labels1[indices]
    labels2 = labels2[indices]

    # Flatten each sequence into a vector
    series1_flat = series1.reshape(n_samples, -1)
    series2_flat = series2.reshape(n_samples, -1)

    # Combine data
    data = np.vstack([series1_flat, series2_flat])
    domain = np.array(['Real'] * n_samples + ['Synthetic'] * n_samples)
    class_labels = np.concatenate([labels1, labels2])

    # Composite label: "Real - class 0", etc.
    composite_labels = np.array([f"{d} - class {c}" for d, c in zip(domain, class_labels)])

    # Create subplots
    if method == 'both':
        fig, axes = plt.subplots(2, 1, figsize=(6, 12))
        methods = ['PCA', 'T-SNE']
    else:
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        methods = [method.upper()]
        axes = [ax]

    for ax, m in zip(axes, methods):
        reducer = PCA(n_components=2) if m == 'PCA' else TSNE(n_components=2, perplexity=10, random_state=42)
        reduced = reducer.fit_transform(data)

        for label in np.unique(composite_labels):
            idx = composite_labels == label
            ax.scatter(reduced[idx, 0], reduced[idx, 1], label=label, alpha=0.6)

        ax.set_title(f"{m} projection", fontsize=13)
        ax.legend(loc='best', fontsize='small')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(save_path, format='pdf')
    plt.show()

    return fig, axes



def compute_dtw_by_class(real_data, fake_data, labels_real, labels_fake, class_names, folder_path='experiments/metrics'):
    """
    computes the dtw os the fake sequences and compare with the real sequences of the same class. The most similiar
    is the dtw of the fake sequence with a real sequence of the same class, the better.                        
    """
    n = min(len(real_data), len(fake_data))
    real_distances_by_class = defaultdict(list)
    fake_distances_by_class = defaultdict(list)

    real_shuffled_data = shuffle_within_classes(real_data, labels_real, seed=42)

    for i in range(n):
        label = labels_real[i]
        if label != labels_fake[i]:
            continue  # ignora se o rótulo da real e da fake não batem (opcional)

        s1 = real_data[i].reshape(1, -1)
        s2 = fake_data[i].reshape(1, -1)
        dist_fake = dtw(s1, s2)[0]

        s2 = real_shuffled_data[i].reshape(1, -1)
        dist_real = dtw(s1, s2)[0]

        real_distances_by_class[label].append(dist_real)
        fake_distances_by_class[label].append(dist_fake)

    #save to a csv file the results
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
    with open(folder_path + f"/{current_time}_dtw_by_class.csv", "w") as f:
        f.write("Class,Real DTW Distances,Fake DTW Distances\n")
        for cls in real_distances_by_class.keys():
            real_dists = real_distances_by_class[cls]
            fake_dists = fake_distances_by_class[cls]
            f.write(f"{class_names[cls]},{np.mean(real_dists)},{np.mean(fake_dists)}\n")
            print(f"Class {class_names[cls]}: Real DTW: {np.mean(real_dists)}, Fake DTW: {np.mean(fake_dists)}")

def shuffle_within_classes(X, Y, seed=None):
    """
    Embaralha as amostras em X_set dentro de cada classe definida em Y_set.
    
    Parâmetros:
        X_set (np.ndarray): Array de amostras com shape (n_samples, ...).
        Y_set (np.ndarray): Array de rótulos com shape (n_samples,).
        seed (int, opcional): Semente para reprodutibilidade do embaralhamento.
    
    Retorna:
        np.ndarray: X_set com amostras embaralhadas dentro de suas respectivas classes.
    """
    if seed is not None:
        np.random.seed(seed)

    X_shuffled = X.copy()

    for cls in np.unique(Y):
        idxs = np.where(Y == cls)[0]
        shuffled_idxs = np.random.permutation(idxs)
        X_shuffled[idxs] = X[shuffled_idxs]

    return X_shuffled

def plot_class_distribution(Y_real, Y_synth=None, class_names=None, title="Distribuição das Classes"):
    """
    Plota a distribuição de classes reais e sintéticas no dataset.

    Parâmetros:
        Y_real (array-like): Labels dos dados reais.
        Y_synth (array-like, opcional): Labels dos dados sintéticos.
        class_names (list, opcional): Nomes legíveis para as classes (usa os valores únicos se None).
        title (str): Título do gráfico.
    """
    Y_real = np.array(Y_real)
    classes = np.unique(Y_real if Y_synth is None else np.concatenate([Y_real, Y_synth]))
    class_names = class_names if class_names is not None else [str(c) for c in classes]

    # Conta amostras reais por classe
    real_counts = [np.sum(Y_real == cls) for cls in classes]

    # Conta amostras sintéticas por classe, se fornecido
    synth_counts = [np.sum(Y_synth == cls) for cls in classes] if Y_synth is not None else [0]*len(classes)

    bar_width = 0.6
    r = np.arange(len(classes))

    plt.figure(figsize=(10, 5))
    plt.bar(r, real_counts, color='skyblue', edgecolor='black', label='Reais')
    if Y_synth is not None:
        plt.bar(r, synth_counts, bottom=real_counts, color='orange', edgecolor='black', label='Sintéticos')

    plt.xticks(r, class_names)
    plt.xlabel("Classe")
    plt.ylabel("Número de Sequências")
    plt.title(title)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_feature_distributions(real_data, fake_data, features_names, real_labels=None, fake_labels=None, n_channels=None, n_bins=50, folder_path='experiments/metrics'):
    """
    Plota a distribuição de cada canal (feature) para real vs sintético.
    
    Args:
        real_data: np.ndarray (N, seq_len, n_channels)
        fake_data: np.ndarray (M, seq_len, n_channels)
        real_labels: np.ndarray (N,) ou None
        fake_labels: np.ndarray (M,) ou None
        n_channels: número de canais a plotar (se None, plota todos)
        n_bins: número de bins para o histograma
    """
    assert real_data.shape[-1] == fake_data.shape[-1], "Número de canais deve coincidir!"
    n_channels = n_channels or real_data.shape[-1]
    
    # Flattens (colapsa seq_len e samples)
    real_flat = real_data.reshape(-1, real_data.shape[-1])
    fake_flat = fake_data.reshape(-1, fake_data.shape[-1])

    fig, axes = plt.subplots(n_channels, 1, figsize=(8, 3 * n_channels))
    if n_channels == 1:
        axes = [axes]

    for i in range(n_channels):
        ax = axes[i]
        ax.hist(real_flat[:, i], bins=n_bins, density=True, alpha=0.6, label='Real')
        ax.hist(fake_flat[:, i], bins=n_bins, density=True, alpha=0.6, label='Synthetic')
        ax.set_title(f'Canal {features_names[i]} - Distribuição')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    # Save plot with current time in the folder path
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
    plot_path = f"{folder_path}/{current_time}_feature_distributions.pdf"
    plt.savefig(plot_path, format='pdf')
    plt.show()

def compare_feature_entropy(real_data, fake_data, n_bins=50, folder_path='experiments/metrics'):
    """
    Calcula e compara a entropia marginal de cada canal (feature).
    
    Args:
        real_data: np.ndarray (N, seq_len, n_channels)
        fake_data: np.ndarray (M, seq_len, n_channels)
        n_bins: número de bins para estimar a densidade
        
    Returns:
        dict com entropias médias e diferença relativa
    """
    assert real_data.shape[-1] == fake_data.shape[-1], "Número de canais deve coincidir!"
    n_channels = real_data.shape[-1]
    
    real_flat = real_data.reshape(-1, n_channels)
    fake_flat = fake_data.reshape(-1, n_channels)
    
    ent_real, ent_fake = [], []
    
    for i in range(n_channels):
        hist_real, _ = np.histogram(real_flat[:, i], bins=n_bins, density=True)
        hist_fake, _ = np.histogram(fake_flat[:, i], bins=n_bins, density=True)

        # remove zeros p/ evitar log(0)
        hist_real = hist_real[hist_real > 0]
        hist_fake = hist_fake[hist_fake > 0]

        ent_real.append(entropy(hist_real))
        ent_fake.append(entropy(hist_fake))
    
    ent_real = np.array(ent_real)
    ent_fake = np.array(ent_fake)
    
    result = {
        "mean_entropy_real": ent_real.mean(),
        "mean_entropy_fake": ent_fake.mean(),
        "relative_difference(%)": 100 * (ent_fake.mean() - ent_real.mean()) / ent_real.mean()
    }
    
    print("📊 Entropia média real:", result["mean_entropy_real"])
    print("📊 Entropia média sintética:", result["mean_entropy_fake"])
    print("Δ Entropia relativa: %.2f%%" % result["relative_difference(%)"])
    
    # Plot comparativo por canal
    plt.figure(figsize=(8,4))
    plt.plot(ent_real, label="Real", marker='o')
    plt.plot(ent_fake, label="Synthetic", marker='x')
    plt.title("Entropia marginal por canal")
    plt.xlabel("Canal")
    plt.ylabel("Entropia (nats)")
    plt.legend()
    plt.grid(True)
    # Save plot to folder
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
    plot_path = f"{folder_path}/{current_time}_entropy_comparison.pdf"
    plt.savefig(plot_path, format='pdf')
    plt.show()
    
    return result

def main():
    ##### Generate data #####
    real_dataset = load_original_dataset(128, is_train=True, attack_only=False, shuffle=True).dataset
    tts_cgan_model_path = "experiments/TTS_APT_CGAN_6_VAR_V_2025_10_06_11_27_36/Model/checkpoint"
    #rcgan_model_path = "RGAN/experiments/settings/dapt2020.txt"
    #time_gan_model_path = "output/TimeGAN/stock/train/weights"

    #generator = RCGAN.SyntheticGenerator(rcgan_model_path, epoch=89)
    #generator = TimeGAN.SyntheticGenerator(time_gan_model_path, real_dataset)
    generator = TTSCGAN.SyntheticGenerator(128, 10, 5, tts_cgan_model_path)

    fake_dataset = generator.generate(real_dataset.Y_test)
    
    ####### DTW #########
    compute_dtw_by_class(real_dataset.X_test, fake_dataset, real_dataset.Y_test, real_dataset.Y_test, real_dataset.classes_names)

    ###### PCA TSE ######
    plot_PCA_TSE(real_dataset.X_test, fake_dataset)

    ##### data distribution ######

    plot_feature_distributions(real_dataset.X_test, fake_dataset, real_dataset.features_names, n_channels=10)

    ##### entropy ######
    compare_feature_entropy(real_dataset.X_test, fake_dataset)

    ####### PLOTS #######
    plot_samples(real_dataset.X_test[50:100], real_dataset.features_names, offset=0, path="images/real_samples.pdf", title="Amostras Reais")
    plot_samples(fake_dataset[50:100], real_dataset.features_names, offset=0, path="images/fake_samples.pdf", title="Amostras Sintéticas (TTS-CGAN)")
    #plot_class_distribution(real_dataset.Y_set, Y_set, class_names=["Benign", "exfiltration", "establish foothold", "lateral movement", "reconnaissance"])
    #####################



if __name__ == "__main__":
    main()