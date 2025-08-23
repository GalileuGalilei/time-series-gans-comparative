import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from classifiers.Classifiers import *
import os
from data.data_loader import DAPT2020
from TSTR import evaluate_cpu_model, evaluate_torch_model, train_cpu_model, train_torch_model

#foram removidas ip, protocol e 'Src Port', pois não são numéricas e não são informações temporais
corr_based_possible_features = ['SYN Flag Count', 'Flow Packets/s', 'Src Port', 'Bwd Packets/s', 'Flow Duration', 'FIN Flag Count', 'Fwd IAT Total', 
                     'Packet Length Min', 'Flow IAT Max', 'Idle Max', 'Fwd IAT Max', 'Idle Mean', 'Flow IAT Std', 'Fwd IAT Std', 'Idle Min', 'Bwd Packet Length Min',
                    'Bwd IAT Total', 'ACK Flag Count', 'Bwd PSH Flags', 'PSH Flag Count', 'Fwd Packets/s', 'Bwd IAT Max', 'Fwd Packet Length Min', 'Bwd IAT Std',
                    'Idle Std', 'Fwd IAT Mean', 'Active Max', 'Bwd IAT Mean', 'Flow IAT Mean', 'Active Mean', 'Active Std', 'Fwd Packet Length Std', 'Flow IAT Min',
                    'Fwd Packet Length Max', 'Active Min', 'Fwd IAT Min', 'Fwd Segment Size Avg', 'Fwd Packet Length Mean', 'RST Flag Count', 'Packet Length Max', 
                    'Bwd Packet Length Std', 'Packet Length Std', 'Bwd Packet Length Max', 'Down/Up Ratio', 'Average Packet Size', 'Packet Length Mean', 
                    'Packet Length Variance', 'Flow Bytes/s', 'Bwd Init Win Bytes', 'Subflow Fwd Packets', 'Total Fwd Packet', 'Fwd Header Length', 
                    'Fwd Act Data Pkts', 'Subflow Bwd Packets', 'Total Bwd packets', 'Bwd Header Length', 'Bwd Segment Size Avg', 'Bwd Packet Length Mean', 
                    'Subflow Bwd Bytes', 'Total Length of Bwd Packet', 'Subflow Fwd Bytes', 'Total Length of Fwd Packet', 'Dst Port', 'CWR Flag Count', 
                    'ECE Flag Count', 'Bwd IAT Min', 'Fwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'URG Flag Count', 'Fwd Bytes/Bulk Avg', 'Fwd Packet/Bulk Avg',
                    'Fwd Bulk Rate Avg', 'Bwd Bytes/Bulk Avg', 'Bwd Packet/Bulk Avg', 'Bwd Bulk Rate Avg', 'FWD Init Win Bytes', 'Fwd Seg Size Min', 'Timestamp']

feta_srd_possible_features =  ['Bwd Packets/s', 'Flow Packets/s', 'Src Port', 'FIN Flag Count', 'SYN Flag Count', 'Flow Duration', 'Fwd IAT Total',
                                'Packet Length Min', 'Flow IAT Max', 'Fwd Packets/s', 'Idle Max', 'Fwd IAT Max', 'ACK Flag Count', 'Idle Mean', 'Flow IAT Std',
                                'Fwd IAT Std', 'Idle Min', 'Bwd IAT Total', 'Bwd Packet Length Min', 'Bwd IAT Max', 'Fwd Packet Length Min', 'Bwd IAT Std',
                                'PSH Flag Count', 'Bwd PSH Flags', 'Idle Std', 'Bwd IAT Mean', 'Fwd IAT Mean', 'Active Max', 'Flow IAT Mean', 'Active Mean',
                                'Active Std', 'Flow IAT Min', 'Active Min', 'Down/Up Ratio', 'Fwd Packet Length Std', 'Fwd Packet Length Max', 'Fwd IAT Min', 
                                'Fwd Segment Size Avg', 'Fwd Packet Length Mean', 'RST Flag Count', 'Packet Length Max', 'Bwd Packet Length Std', 'Dst Port', 
                                'Packet Length Std', 'Bwd Packet Length Max', 'Flow Bytes/s', 'Average Packet Size', 'Packet Length Mean', 'Bwd Init Win Bytes', 
                                'Packet Length Variance', 'Bwd Packet Length Mean', 'Bwd Segment Size Avg', 'Subflow Fwd Packets', 'Total Fwd Packet', 
                                'Fwd Header Length', 'Fwd Act Data Pkts', 'Subflow Bwd Packets', 'Total Bwd packets', 'Bwd Header Length', 'Subflow Bwd Bytes', 
                                'Total Length of Bwd Packet', 'Subflow Fwd Bytes', 'Total Length of Fwd Packet', 'Bwd IAT Min', 'ECE Flag Count', 'CWR Flag Count', 
                                'Fwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'URG Flag Count', 'Fwd Bytes/Bulk Avg', 'Fwd Packet/Bulk Avg', 
                                'Fwd Bulk Rate Avg', 'Bwd Bytes/Bulk Avg', 'Bwd Packet/Bulk Avg', 'Bwd Bulk Rate Avg', 'FWD Init Win Bytes', 'Fwd Seg Size Min', "Timestamp"]

original_features = ['SYN Flag Count', 'Src Port', 'Fwd Packets/s', 'Flow Packets/s', 'Bwd Packets/s', 'ACK Flag Count', 'FIN Flag Count', 'Flow Bytes/s', 'Timestamp']

def calculate_features_score():

    full_dataset = DAPT2020("data/output.csv", feta_srd_possible_features, 'Stage', 30, is_train=True, attack_only=False, shuffle=True, expand=True)

    #f1 score para cada classificador dependendo do número de features, ordenadas por correlação absoluta
    header = "num_features,RF,SVM,LSTM,Transformer,Average\n" 

    for num_features in range(1, len(feta_srd_possible_features) + 1):
        F1_scores = []
        classifiers = [RandomForestClassifierModel(50),
                    SVMClassifier(),
                    LSTMClassifier(num_features, 30, 64, 5),
                    TransformerClassifier(num_features, 30, 5)]
        print(f"Testing with {num_features} features...")
        for classifier in classifiers:
            
            # Treina o modelo
            if classifier.is_torch_model():
                model = train_torch_model(full_dataset.X_train_set[:, :num_features, :], full_dataset.Y_train_set, classifier.copy())
                _,_,f1_score = evaluate_torch_model(full_dataset.X_test_set[:, :num_features, :], full_dataset.Y_test_set, model).calculate_weighted_metrics()
            else:
                model = train_cpu_model(full_dataset.X_train_set[:, :num_features, :], full_dataset.Y_train_set, classifier.copy())
                _,_,f1_score = evaluate_cpu_model(full_dataset.X_test_set[:, :num_features, :], full_dataset.Y_test_set, model).calculate_weighted_metrics()
            
            F1_scores.append(f1_score)

        average_f1 = sum(F1_scores) / len(F1_scores)
        header += f"{num_features},{F1_scores[0]:.4f},{F1_scores[1]:.4f},{F1_scores[2]:.4f},{F1_scores[3]:.4f},{average_f1:.4f}\n"

    # Salva os resultados em um arquivo CSV
    with open("experiments/features_tuning_results.csv", "w") as f:
        print("Saving results to features_tuning_results.csv")
        f.write(header)
'''
def fine_tuning(model):
    """
    target_ratios: dicionário com {classe: proporção desejada no total final}
    """
    grow_steps = [0.25, 0.3, 0.35, 0.4]
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


    model_path = "logs/TTS_APT_CGAN_6_VAR_V4_head4_128_160/Model/checkpoint"
    features_to_train = ['Bwd Packets/s', 'Flow Packets/s', 'Src Port', 'Protocol', 'FIN Flag Count', 'SYN Flag Count', 'Timestamp']
    data_path = "data/output.csv"
    seq_len = 30
    scores = []

    best_target_ratio = {}
    best_score = 0

    test_set = load_original_dataset(is_train=False, attack_only=False)
    train_set = DAPT2020(data_path, list(features_to_train), "Stage", seq_len, is_train=True)
    generator = TTSCGAN.SyntheticGenerator(seq_len=seq_len, num_channels=6, num_classes=5, model_path=model_path)

    for target_ratio in target_ratios:
        print(f"Target ratio: {target_ratio}")
        train_set_increased = generate_semi_syntetic_dataset(train_set, generator, target_ratio)

        # roda o modelo
        trained_model = train_cpu_model(train_set_increased.dataset.X_train_set, train_set_increased.dataset.Y_train_set, model.copy())
        #retorna f1 score
        recort = evaluate_cpu_model(test_set.dataset.X_test_set, test_set.dataset.Y_test_set, trained_model)
        _, _, f1 = recort.calculate_weighted_metrics()
        print(f"F1 Score: {f1}")
        scores.append(f1)

        if f1 > best_score:
            best_score = f1
            best_target_ratio = target_ratio
    print(f"Melhor target ratio: {best_target_ratio} com f1 score: {best_score}")
'''

if __name__ == "__main__":

    if not os.path.exists("experiments/features_tuning_results.csv"):
        print("Calculating features score...")
        calculate_features_score()

    df = pd.read_csv("experiments/features_tuning_results.csv")

    # Plota os resultados
    plt.figure(figsize=(10, 6))
    num_features = df['num_features']
    plt.plot(num_features, df['RF'], label='RF', marker='o')
    plt.plot(num_features, df['SVM'], label='SVM', marker='o')
    plt.plot(num_features, df['LSTM'], label='LSTM', marker='o')
    plt.plot(num_features, df['Transformer'], label='Transformer', marker='o')
    plt.plot(num_features, df['Average'], label='Average', marker='o')
    plt.title('F1 Score vs Number of Features')
    plt.xlabel('Number of Features')
    plt.ylabel('F1 Score')
    plt.xticks(num_features)
    plt.legend()
    plt.grid()
    plt.savefig("images/features_tuning_plot.pdf", format="pdf")
    plt.show()




