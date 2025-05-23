from RGAN import data_utils
from sklearn.calibration import label_binarize
from data.DataLoader import *
from TTSCGAN.generate_data import *
from classifiers import Classifiers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from matplotlib import pyplot as plt

model_path = "logs/TTS_APT_CGAN_OITO_VAR_IMPR7/Model/checkpoint"

def load_data(is_train, attack_only=False):
    features = ['SYN Flag Count', 'Src Port', 'Fwd Packets/s', 'Flow Packets/s', 'Bwd Packets/s', 'ACK Flag Count', 'FIN Flag Count', 'Flow Bytes/s', 'Timestamp']
    label_column = 'Stage'
    seq_len = 30
    filename = "data/output.csv"
    
    #ja embaralhado por padrao
    data_set = load_and_preprocess_data(filename, features, label_column, seq_len, is_train=is_train, attack_only=attack_only, shuffle=True)

    return DataLoader(data_set, batch_size=64, shuffle=True)

def recreate_dataset_RCGAN(shuffle=True, seed=22):
    synt_data, synt_labels, _, _ = data_utils.generate_synthetic("dapt2020", 59)
    synt_data = np.transpose(synt_data, (0, 2, 1))  # Transpose to (batch_size, seq_len, num_channels)
    synt_data = np.expand_dims(synt_data, axis=2)

    #revert one hot encoding
    synt_labels = np.argmax(synt_labels, axis=1)

    # Create a DataFrame from the synthetic data
    train_set = SynthDataset(synt_data, synt_labels)
    # Shuffle the dataset if required
    if shuffle:
        np.random.seed(seed)
        indices = np.arange(len(train_set))
        np.random.shuffle(indices)
        train_set.X_set = train_set.X_set[indices]
        train_set.Y_set = train_set.Y_set[indices]

    return DataLoader(train_set, batch_size=64, shuffle=True)

#sempre sera para treino
def generate_data(attack_increase=False):
    features = ['SYN Flag Count', 'Src Port', 'Fwd Packets/s', 'Flow Packets/s', 'Bwd Packets/s', 'ACK Flag Count', 'FIN Flag Count', 'Flow Bytes/s', 'Timestamp']
    data_path = "data/output.csv"
    seq_len = 30
    num_classes = 5
    num_channels = 8
    filename = "data/output.csv"

    trainable_features = ['SYN Flag Count', 'Src Port', 'Fwd Packets/s', 'Flow Packets/s', 'Bwd Packets/s', 'ACK Flag Count', 'FIN Flag Count', 'Flow Bytes/s', 'Timestamp']
    seq_len = 30

    train_set = load_and_preprocess_data(data_path, list(trainable_features), "Stage", seq_len, is_train=True, shuffle=True)
    y_train = train_set.Y_set
    x_train = train_set.X_set

    num_classes = max(y_train) + 1
    num_channels = x_train.shape[1]
    gen_net = load_model_generator(seq_len=seq_len, num_channels=num_channels, num_classes=num_classes, model_path=model_path)

    if attack_increase:
        #target_ratio = {0: 0, 1: 0.1, 2: 0.1, 3: 0.4, 4: 0.1}
        #target_ratio = {0: 0, 1: 0.1, 2: 0.2, 3: 0.2, 4: 0.1}
        #target_ratio = {0: 0, 1: 0.1, 2: 0.1, 3: 0.2, 4: 0.1}
        #target_ratio = {0: 0, 1: 0.1, 2: 0.3, 3: 0.3, 4: 0.1}
        target_ratio = {0: 0, 1: 0.3, 2: 0.2, 3: 0.2, 4: 0.2}
        return DataLoader(recreate_balanced_dataset(train_set, gen_net, target_ratio), batch_size=64, shuffle=True)
    
    #terá o mesmo numero de amostras que o original
    return DataLoader(recreate_dataset(filename, model_path, features, seq_len), batch_size=16, shuffle=True)

def generate_mixed_data(new_samples_prop=0.5, attack_only=False):
    features = ['SYN Flag Count', 'Src Port', 'Fwd Packets/s', 'Flow Packets/s', 'Bwd Packets/s', 'ACK Flag Count', 'FIN Flag Count', 'Flow Bytes/s', 'Timestamp']
    gen_model_path = model_path
    seq_len = 30
    filename = "data/output.csv"

    dataset = create_mixed_dataset(filename, gen_model_path, features, seq_len, new_samples_prop, attack_only)

    return DataLoader(dataset, batch_size=16, shuffle=True)

def train_cpu_model(train_loader, model):
    print("Training Classifier")
    
    # Inicializa a função de perda e o otimizador
    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.fit(train_loader.dataset.X_set, train_loader.dataset.Y_set)
    print("Training completed")
    return model

def evaluate_cpu_model(test_data, model):
    prediction = model.predict(test_data.dataset.X_set)
    all_preds = np.array(prediction)
    all_labels = np.array(test_data.dataset.Y_set)

    # Calcula métricas de desempenho
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Matriz de Confusão
    conf_matrix = confusion_matrix(all_labels, all_preds)
    false_positives = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
    false_negatives = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
    # Exibe os resultados
    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("Matriz de Confusão:")
    print(conf_matrix)
    print(f"Falsos Positivos por classe: {false_positives}")
    print(f"Falsos Negativos por classe: {false_negatives}")

    return f1
    

def train_torch_model(train_loader, model):

    print("Training Classifier")
    # Initialize the classifier
    # Define o dispositivo (GPU se disponível, caso contrário, CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move o modelo para GPU (caso necessário)

    epochs = 10

    # Inicializa a função de perda e o otimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        # Loop de treinamento
        for i, (X_batch, y_batch) in enumerate(train_loader):
            # Move os dados para o dispositivo correto
            X_batch = X_batch.float().to(device)  # Converte para float32 e envia para GPU
            y_batch = y_batch.long().to(device)   # Converte para long e envia para GPU

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass e otimização
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    print("Training completed")
    return model

def evaluate_torch_model(test_loader, model):
    print("Evaluating Classifier")
    seq_len = 30

    # Load the model
    #hidden_dim = 64
    #n_classes = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define o modelo em modo de avaliação
    model.eval()

    # Move o modelo para GPU (caso necessário)
    model.to(device)

    # Listas para armazenar os resultados
    all_preds = []
    all_labels = []

    # Loop pelos dados de teste
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            # Move os dados para GPU
            X_batch = X_batch.float().to(device)
            y_batch = y_batch.long().to(device)

            # Forward pass
            outputs = model(X_batch)

            # Obtém as previsões (índice da classe com maior probabilidade)
            preds = torch.argmax(outputs, dim=1)

            # Armazena os resultados
            all_preds.extend(preds.cpu().numpy())  # Move para CPU para processamento
            all_labels.extend(y_batch.cpu().numpy())

    # Converte listas para arrays NumPy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calcula métricas de desempenho
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Matriz de Confusão
    conf_matrix = confusion_matrix(all_labels, all_preds)
    false_positives = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
    false_negatives = conf_matrix.sum(axis=1) - np.diag(conf_matrix)

    # Exibe os resultados
    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("Matriz de Confusão:")
    print(conf_matrix)
    print(f"Falsos Positivos por classe: {false_positives}")
    print(f"Falsos Negativos por classe: {false_negatives}")

    return f1

def find_best_target_ratios(model, target_ratios):
    """
    target_ratios: dicionário com {classe: proporção desejada no total final}
    """
    model_path = "logs/TTS_APT_CGAN_OITO_VAR_IMPR7/Model/checkpoint"
    trainable_features = ['SYN Flag Count', 'Src Port', 'Fwd Packets/s', 'Flow Packets/s', 'Bwd Packets/s', 'ACK Flag Count', 'FIN Flag Count', 'Flow Bytes/s', 'Timestamp']
    data_path = "data/output.csv"
    seq_len = 30
    scores = []

    test_set = load_data(is_train=False, attack_only=False)
    train_set = load_and_preprocess_data(data_path, list(trainable_features), "Stage", seq_len, is_train=True, shuffle=True)
    y_train = train_set.Y_set
    x_train = train_set.X_set

    num_classes = max(y_train) + 1
    num_channels = x_train.shape[1]
    gen_net = load_model_generator(seq_len=seq_len, num_channels=num_channels, num_classes=num_classes, model_path=model_path)

    for target_ratio in target_ratios:
        print(f"Target ratio: {target_ratio}")
        train_set_increased = DataLoader(recreate_balanced_dataset(train_set, gen_net, target_ratio), batch_size=64, shuffle=True)

        # roda o modelo
        trained_model = train_cpu_model(train_set_increased, model.copy())
        #retorna f1 score
        f1 = evaluate_cpu_model(test_set, trained_model)
        scores.append(f1)

        
    # Plota os resultados
    plt.plot(range(len(target_ratios)), scores, marker='o')
    plt.xticks(range(len(target_ratios)), [str(r) for r in target_ratios], rotation=45)
    plt.xlabel('Target Ratios')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Target Ratios')
    plt.grid()
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    #data = load_data(is_train=True, attack_only=False)
    data = recreate_dataset_RCGAN(shuffle=True, seed=22)
    #data = generate_data(attack_increase=False)
    #data = generate_data(attack_increase=True)

    #trained_model = train_torch_model(data, Classifiers.TransformerClassifier(8, 30, 5))    
    #evaluate_torch_model(load_data(is_train=False, attack_only=False), trained_model)

    #trained_model = train_torch_model(data, Classifiers.LSTMClassifier(30, 64, 5))
    #evaluate_torch_model(load_data(is_train=False, attack_only=False), trained_model)

    #trained_model = train_cpu_model(data, Classifiers.SVMClassifier())
    #evaluate_cpu_model(load_data(is_train=False, attack_only=False), trained_model)

    trained_model = train_cpu_model(data, Classifiers.RandomForestClassifierModel(50))
    evaluate_cpu_model(load_data(is_train=False, attack_only=False), trained_model)


    # grow_steps = [0.1, 0.2, 0.3, 0.4]
    # target_ratio_best = {0: 0, 1: 0.4, 2: 0.2, 3: 0.3, 4: 0.4}
    # target_ratios = []
    # i = 0
    # while i < len(grow_steps)**4:
    #     target_ratios.append(
    #         {0: 0, 1: grow_steps[i%len(grow_steps)],
    #          2: grow_steps[(i//len(grow_steps))%len(grow_steps)],
    #          3: grow_steps[(i//(len(grow_steps)**2))%len(grow_steps)],
    #          4: grow_steps[(i//(len(grow_steps)**3))%len(grow_steps)]})
    #     i += 1

    # model = Classifiers.RandomForestClassifierModel(50)
    # find_best_target_ratios(model, target_ratios)