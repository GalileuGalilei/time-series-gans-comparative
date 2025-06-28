from data_utils import *
from collections import defaultdict
from tslearn.metrics import dtw
from TimeGAN import SyntheticGenerator

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def plot_samples(data, features_names, labels=None, offset=0, path=None, title="Samples"):
    fig, axs = plt.subplots(2, 5, figsize=(20, 5))
    fig.suptitle(title, fontsize=15)

    # Definição de cores
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    num_samples = data.shape[0]
    num_classes = len(features_names)

    for i in range(2):
        for j in range(5):
            sample_idx = i * 5 + j + offset
            if sample_idx >= num_samples:
                break  # Evita acessar índices fora do alcance de 'data'
            for k in range(1, num_classes):
                if k >= len(colors):
                    break  # Evita acessar índices fora do alcance de 'colors'
                axs[i, j].plot(data[sample_idx, k, 0, :], color=colors[k], label=features_names[k])

            # Alteração da cor de fundo com base no rótulo
            if labels is not None:
                if sample_idx < len(labels):
                    axs[i, j].set_facecolor('white' if labels[sample_idx] == 0 else 'red')

    # Criação da legenda
    handles = [plt.Line2D([0], [0], color=colors[k], lw=2) for k in range(1, num_classes) if k < len(colors)]
    fig.legend(handles, features_names[1:num_classes], loc='upper right', fontsize=12)

    if path:
        plt.savefig(path)

def plot_PCA_TSE(series1, series2, method='both'):
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
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        methods = ['PCA', 'T-SNE']
    else:
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        methods = [method.upper()]
        axes = [ax]

    for ax, m in zip(axes, methods):
        if m == 'PCA':
            reducer = PCA(n_components=2)
        elif m == 'T-SNE':
            reducer = TSNE(n_components=2, perplexity=30, random_state=42)
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
    plt.suptitle("Comparação entre Séries Temporais Reais e Sintéticas")
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    # Save the figure
    plt.savefig("images/plot_pca_tse.png")
    plt.show()
    return fig, axes

def compute_dtw_by_class(real_data, fake_data, labels_real, labels_fake):
    """
    real_data: np.array de shape (N, C, 1, T)
    fake_data: np.array de shape (N, C, 1, T)
    labels_real: array/list com shape (N,) - rótulos das amostras reais
    labels_fake: array/list com shape (N,) - rótulos das amostras geradas
    """
    n = min(len(real_data), len(fake_data))
    distances_by_class = defaultdict(list)

    for i in range(n):
        label = labels_real[i]
        if label != labels_fake[i]:
            continue  # ignora se o rótulo da real e da fake não batem (opcional)

        dist = dtw(real_data[i].reshape(1, -1), fake_data[i].reshape(1, -1))
        distances_by_class[label].append(dist)

    print("DTW por classe:")
    for label, dists in distances_by_class.items():
        mean = np.mean(dists)
        std = np.std(dists)
        print(f"Classe {label}: Média = {mean:.4f}, Desvio = {std:.4f}")

    return distances_by_class

def shuffle_within_classes(dataset, seed=None):
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

    X_set = dataset.X_set
    Y_set = dataset.Y_set
    X_shuffled = X_set.copy()

    for cls in np.unique(Y_set):
        idxs = np.where(Y_set == cls)[0]
        shuffled_idxs = np.random.permutation(idxs)
        X_shuffled[idxs] = X_set[shuffled_idxs]

    dataset.X_set = X_shuffled
    dataset.Y_set = Y_set
    return dataset

import numpy as np
import matplotlib.pyplot as plt

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


def main():
    tts_cgan_model_path = "logs/TTS_APT_CGAN_16_VAR_2025_06_27_23_43_06/Model/checkpoint"
    rcgan_model_path = "RGAN/experiments/settings/dapt2020.txt"
    #fake_dataset = recreate_dataset(data_path, model_path, list(features_names), seq_len, shuffle=True)
    generator = TTSCGAN.SyntheticGenerator(30, 6, 5, tts_cgan_model_path)
    #generator = RCGAN.SyntheticGenerator(rcgan_model_path, epoch=89)
    #generator = SyntheticGenerator("C:/Users/Alfredo/source/repos/cyberdata-improvement-tts-cgan/output/TimeGAN/stock/train/weights", real_dataset)
    real_dataset = load_original_dataset(is_train=True, attack_only=False, Shuffle=False, expand=True).dataset
    fake_dataset = generator.generate(real_dataset.Y_test_set)
    
    #real_dataset_shuffled = load_and_preprocess_data(data_path, list(features_names), "Stage", seq_len, is_train=True, shuffle=True, seed=22)
    #real_dataset_shuffled = shuffle_within_classes(real_dataset_shuffled)

    #### !!!!!!!!!!!!!!!!!! ####
    # o dataset original comparado com ele mesmo não possui similariedade igual à 1.0 como normalmente se espera,
    # isso se deve provavelmente à grande variação entre os dados.
    # o dataset sintético possui similaridade um pouco maior, pois é uma generalização
    #então na média ele é mais próximo do real, por isso ultrapassa um pouco o original.
    #### !!!!!!!!!!!!!!!!!! ####

    #dynamic time warping
    #_ = compute_dtw_by_class(real_dataset.X_train_set, fake_dataset.X_train_set, real_dataset.Y_train_set, fake_dataset.Y_train_set)

    plot_PCA_TSE(real_dataset.X_test_set, fake_dataset)
    plot_samples(real_dataset.X_test_set[20:40], real_dataset.features_names, None, offset=0, path="images/real_samples.png", title="Dados reais processados")
    plot_samples(fake_dataset, real_dataset.features_names, None, offset=0, path="images/synthetic_samples.png", title="Dados sintéticos gerados")

    # Calculate Cosine Similarity
    #mean_sim, std_sim = compute_cosine_similarity(real_dataset_shuffled.X_set, fake_dataset.X_set, n_samples=100)
    #print(f"Mean Cosine Similarity: {mean_sim}")
    #print(f"Standard Deviation of Cosine Similarity: {std_sim}")

    #plot_class_distribution(real_dataset.Y_set, fake_dataset.dataset.Y_set, class_names=["Benign", "exfiltration", "establish foothold", "lateral movement", "reconnaissance"])



if __name__ == "__main__":
    main()