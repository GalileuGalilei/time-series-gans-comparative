from sklearn.calibration import label_binarize
from data.DataLoader import *
from tts_cgan.generate_data import *
from ML_Models import Classifiers
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_curve, precision_score, recall_score, f1_score, confusion_matrix
from matplotlib import pyplot as plt

def load_data(is_train):
    features = ['SYN Flag Count', 'Src Port', 'Fwd Packets/s', 'Flow Packets/s', 'Bwd Packets/s', 'Flow Bytes/s', 'Timestamp']
    label_column = 'Stage'
    seq_len = 30
    filename = "data/output.csv"
    
    #ja embaralhado por padrao
    data_set = load_and_preprocess_data(filename, features, label_column, seq_len, is_train=is_train)
    return DataLoader(data_set, batch_size=16, shuffle=False, drop_last=False)

#sempre sera para treino
def generate_data(is_balanced):
    features = ['SYN Flag Count', 'Src Port', 'Fwd Packets/s', 'Flow Packets/s', 'Bwd Packets/s', 'Flow Bytes/s', 'Timestamp']
    gen_model_path = "logs/TTS_APT_CGAN_IMPROV/Model/checkpoint"
    seq_len = 30
    num_classes = 5
    num_channels = 6
    filename = "data/output.csv"

    if is_balanced:
        return DataLoader(recreate_balanced_dataset(gen_model_path, features, seq_len, 3000, num_classes, num_channels), batch_size=16, shuffle=True, drop_last=True)
        #return DataLoader(recreate_increased_dataset(filename, gen_model_path, features, seq_len, 8), batch_size=16, shuffle=True, drop_last=True)
    
    #terá o mesmo numero de amostras que o original
    return DataLoader(recreate_dataset(filename, gen_model_path, features, seq_len), batch_size=16, shuffle=True, drop_last=True)

def generate_mixed_data(proportion=0.5):
    features = ['SYN Flag Count', 'Src Port', 'Fwd Packets/s', 'Flow Packets/s', 'Bwd Packets/s', 'Flow Bytes/s', 'Timestamp']
    gen_model_path = "logs/TTS_APT_CGAN_IMPROV/Model/checkpoint"
    seq_len = 30
    num_classes = 5
    num_channels = 6
    filename = "data/output.csv"

    multiply = int(1/proportion) - 1

    return DataLoader(create_mixed_dataset(filename, gen_model_path, features, seq_len, multiply), batch_size=16, shuffle=True, drop_last=False)

def train(train_loader):

    print("Training Classifier")
    # Initialize the classifier
    # Define o dispositivo (GPU se disponível, caso contrário, CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Inicializa o modelo e move para a GPU
    hidden_dim = 64
    seq_len = 30
    n_classes = 5
    epochs = 8
    model = Classifiers.LSTMClassifier(seq_len, seq_len, hidden_dim, n_classes).to(device)
    #model = Classifiers.CNNCassifier().to(device)

    # Inicializa a função de perda e o otimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Cria o DataLoader
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

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
                print(f"Epoch [{epoch+1}/5], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    print("Training completed, saving model...")
    torch.save(model.state_dict(), "LSTM_Classfier.pth")
    #torch.save(model.state_dict(), "CNN_Classfier.pth")

def plot_multiclass_pr_curves(model, dataloader, n_classes, device, class_names=None):
    model.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            # Move os dados para GPU
            X_batch = X_batch.float().to(device)
            y_batch = y_batch.long().to(device)

            probs = model(X_batch)
            all_probs.append(probs.cpu())
            all_targets.append(y_batch.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # Binarize labels for OvR
    y_true_bin = label_binarize(all_targets, classes=list(range(n_classes)))

    plt.figure(figsize=(10, 8))

    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], all_probs[:, i])
        avg_prec = average_precision_score(y_true_bin[:, i], all_probs[:, i])
        label = class_names[i] if class_names else f"Class {i}"
        plt.plot(recall, precision, lw=2, label=f"{label} (AP={avg_prec:.2f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curva de Precision-Recall por classe")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def evaluate_precision_recall():
    print("Evaluating Classifier")
    seq_len = 30
    test_loader = load_data(is_train=False)

    # Load the model
    hidden_dim = 64
    n_classes = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Classifiers.LSTMClassifier(seq_len, seq_len, hidden_dim, n_classes).to(device)
    model.load_state_dict(torch.load("LSTM_Classfier.pth"))

    # Define o modelo em modo de avaliação
    model.eval()

    # Move o modelo para GPU (caso necessário)
    model.to(device)

    plot_multiclass_pr_curves(model, test_loader, n_classes, device)

def evaluate():
    print("Evaluating Classifier")
    seq_len = 30
    test_loader = load_data(is_train=False)

    # Load the model
    hidden_dim = 64
    n_classes = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Classifiers.LSTMClassifier(seq_len, seq_len, hidden_dim, n_classes).to(device)
    model.load_state_dict(torch.load("LSTM_Classfier.pth"))
    #model = Classifiers.CNNCassifier().to(device)
    #model.load_state_dict(torch.load("CNN_Classfier.pth"))

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

if __name__ == "__main__":
    train(load_data(is_train=True))
    #evaluate()
    evaluate_precision_recall()


    #train_loader = generate_data(is_balanced=False) #numeo de amostras aqui n importa
    #train_loader = generate_data(is_balanced=True)
    #train_loader = generate_mixed_data(0.5)
    #train_loader = load_data(is_train=True)