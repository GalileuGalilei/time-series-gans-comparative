from RGAN import data_utils
from metrics import plot_PCA_TSE
from matplotlib import pyplot as plt

def main():
    
    synt_data, synt_labels, real_data, real_labels = data_utils.generate_synthetic("dapt2020", 59)
    fig, axes = plot_PCA_TSE(synt_data, real_data, method='both')
    save_path = "images/RGAN_PCA.png"
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    

if __name__ == "__main__":
    main()