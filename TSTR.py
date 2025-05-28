from RGAN import RCGAN_generator as RCGANGEN
from data.DataLoader import *
from classifiers import Classifiers
from TTSCGAN.generate_data import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix



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

def load_original_dataset(is_train, attack_only=False):
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
        train_set.X_train_set = train_set.X_train_set[indices]
        train_set.Y_train_set = train_set.Y_train_set[indices]

    return DataLoader(train_set, batch_size=64, shuffle=True)

def generate_semi_syntetic_dataset(original_dataset, generator : IGenerator, target_ratios):
    """
    target_ratios: dicionário com {classe: proporção desejada no total final}
    """
    y_train = original_dataset.Y_train_set
    x_train = original_dataset.X_train_set

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

    print(f"Total de amostras sintéticas geradas: {len(fake_sigs)}")

    x_train = np.concatenate((x_train, fake_sigs), axis=0)
    y_train = np.concatenate((y_train, y_synthetic), axis=0)

    return DataLoader(SynthDataset(x_train, y_train), batch_size=64, shuffle=True)

def generate_syntetic_dataset(orignal_dataset, generator : IGenerator):
    """
    Recreate the dataset using the generator model following the order of the original dataset labels
    """
    y_train = orignal_dataset.Y_train_set

    fake_sigs = generator.generate(y_train)
    orignal_dataset = SynthDataset(fake_sigs, y_train)

    return DataLoader(orignal_dataset, batch_size=64, shuffle=True)


def train_cpu_model(X_set, Y_set, model):
    print("Training Classifier")
    
    # Inicializa a função de perda e o otimizador
    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.fit(X_set, Y_set)
    print("Training completed")
    return model

def evaluate_cpu_model(X_set, Y_set, model):
    prediction = model.predict(X_set)
    all_preds = np.array(prediction)
    all_labels = np.array(Y_set)

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
    train_set = load_and_preprocess_data(data_path, list(trainable_features), "Stage", seq_len, is_train=True, shuffle=True)
    generator = TTSCGANGenerator(seq_len=seq_len, num_channels=len(trainable_features)-1, num_classes=5, model_path=model_path)
    y_train = train_set.Y_train_set
    x_train = train_set.X_train_set

    num_classes = max(y_train) + 1
    num_channels = x_train.shape[1]

    for target_ratio in target_ratios:
        print(f"Target ratio: {target_ratio}")
        train_set_increased = generate_semi_syntetic_dataset(train_set, generator, target_ratio)

        # roda o modelo
        trained_model = train_cpu_model(train_set_increased.dataset.X_train_set, train_set_increased.dataset.Y_train_set, model.copy())
        #retorna f1 score
        f1 = evaluate_cpu_model(test_set.dataset.X_test_set, test_set.dataset.Y_test_set, trained_model)
        scores.append(f1)

        if f1 > best_score:
            best_score = f1
            best_target_ratio = target_ratio
    print(f"Melhor target ratio: {best_target_ratio} com f1 score: {best_score}")

        

    

if __name__ == "__main__":
    seq_len = 30
    num_channels = 8
    num_classes = 5
    target_ratios = {0: 0, 1: 0.3, 2: 0.2, 3: 0.2, 4: 0.2}
    tts_cgan_model_path = "TTSCGAN/logs/TTS_APT_CGAN_OITO_VAR_IMPR7/Model/checkpoint"
    original_dataset = load_original_dataset(is_train=True, attack_only=False)
    
    #generator = TTSCGANGenerator(seq_len=30, num_channels=8, num_classes=5, model_path=tts_cgan_model_path)
    generator = RCGANGEN.SyntGenerator(model_path="RGAN/experiments/settings/dapt2020.txt", epoch=59)
    data = generate_semi_syntetic_dataset(original_dataset.dataset, generator, target_ratios)

    #trained_model = train_torch_model(data, Classifiers.TransformerClassifier(8, 30, 5))    
    #evaluate_torch_model(load_original_dataset(is_train=False, attack_only=False), trained_model)

    #trained_model = train_torch_model(data, Classifiers.LSTMClassifier(30, 64, 5))
    #evaluate_torch_model(load_data(is_train=False, attack_only=False), trained_model)

    #trained_model = train_cpu_model(data, Classifiers.SVMClassifier())
    #evaluate_cpu_model(load_data(is_train=False, attack_only=False), trained_model)

    trained_model = train_cpu_model(data.dataset.X_train_set, data.dataset.Y_train_set, Classifiers.RandomForestClassifierModel(50))
    evaluate_cpu_model(original_dataset.dataset.X_test_set, original_dataset.dataset.Y_test_set, trained_model)

    #fine_tuning(Classifiers.SVMClassifier())