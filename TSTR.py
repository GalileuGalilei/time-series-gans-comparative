from data_utils import *
from classifiers.Classifiers import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_cpu_model(X_set, Y_set, model):
    print("Training Classifier")

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

    return accuracy, precision, recall, f1
    

def train_torch_model(X_train_set, Y_train_set, model):

    print("Training Classifier")
    # Initialize the classifier
    # Define o dispositivo (GPU se disponível, caso contrário, CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move o modelo para GPU (caso necessário)

    epochs = 10
    batch_size = 64

    # Inicializa a função de perda e o otimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        # Loop de treinamento
        for i in range(0, len(X_train_set), batch_size):
            # Obtém o lote de dados
            X_batch = X_train_set[i:i+batch_size]
            y_batch = Y_train_set[i:i+batch_size]
            # Converte numpy arrays para tensores e move para o dispositivo correto
            X_batch = torch.from_numpy(X_batch).float().to(device)  # Converte para float32 e envia para GPU
            y_batch = torch.from_numpy(y_batch).long().to(device)   # Converte para long e envia para GPU
            

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass e otimização
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    print("Training completed")
    return model

def evaluate_torch_model(X_test_set, Y_test_set, model):
    print("Evaluating Classifier")

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

    batch_size = 64

    # Loop pelos dados de teste
    with torch.no_grad():
        for i in range(0, len(X_test_set), batch_size):
            # Obtém o lote de dados
            X_batch = X_test_set[i:i+batch_size]
            y_batch = Y_test_set[i:i+batch_size]
            # Move os dados para GPU
            X_batch = torch.from_numpy(X_batch).float().to(device)
            y_batch = torch.from_numpy(y_batch).long().to(device)

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

    return accuracy, precision, recall, f1

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
    train_set = load_and_preprocess_data(data_path, list(trainable_features), "Stage", seq_len, is_train=True)
    generator = TTSCGANGenerator(seq_len=seq_len, num_channels=len(trainable_features)-1, num_classes=5, model_path=model_path)

    for target_ratio in target_ratios:
        print(f"Target ratio: {target_ratio}")
        train_set_increased = generate_semi_syntetic_dataset(train_set, generator, target_ratio)

        # roda o modelo
        trained_model = train_cpu_model(train_set_increased.dataset.X_train_set, train_set_increased.dataset.Y_train_set, model.copy())
        #retorna f1 score
        accuracy_score, precision, recall, f1 = evaluate_cpu_model(test_set.dataset.X_test_set, test_set.dataset.Y_test_set, trained_model)
        scores.append(f1)

        if f1 > best_score:
            best_score = f1
            best_target_ratio = target_ratio
    print(f"Melhor target ratio: {best_target_ratio} com f1 score: {best_score}")

def experiments_battery(generators : list[IGenerator], classifiers : list[IClassifier], original_dataset : DataLoader, save_path: str = "experiments/"):
    """
    Roda uma bateria de experimentos com diferentes geradores e salva os resultados.
    """

    #verifica se o caminho de salvamento existe, se não, cria
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    class ExperimentResult:
        def __init__(self, generator_name, classifier_name, data_type, accuracy, precision, recall, f1):
            self.generator_name = generator_name
            self.classifier_name = classifier_name
            self.data_type = data_type
            self.accuracy = accuracy
            self.precision = precision
            self.recall = recall
            self.f1 = f1

        def __str__(self):
            return (f"Generator: {self.generator_name}, Classifier: {self.classifier_name}, "
                    f"Data Type: {self.data_type}, Accuracy: {self.accuracy:.4f}, "
                    f"Precision: {self.precision:.4f}, Recall: {self.recall:.4f}, F1: {self.f1:.4f}")

    results = []

    for gen in generators:
        for clf in classifiers:
            print(f"Running experiment with generator: {gen.get_name} and classifier: {clf.get_name}")
            # Gera o dataset sintético
            data_synt = generate_syntetic_dataset(original_dataset.dataset, gen)

            # Como o modelo é treinado em CPU ou GPU, escolhe a função de treinamento e avaliação apropriada
            (train_function, eval_function) = (train_torch_model, evaluate_torch_model) if clf.is_torch_model() else (train_cpu_model, evaluate_cpu_model)

            # Treina e avalia o modelo
            trained_model = train_function(data_synt.dataset.X_train_set, data_synt.dataset.Y_train_set, clf.copy())
            accuracy_score, precision, recall, f1 = eval_function(original_dataset.dataset.X_test_set, original_dataset.dataset.Y_test_set, trained_model)

            synt_result = ExperimentResult(
                generator_name=gen.get_name,
                classifier_name=clf.get_name(),
                data_type="synthetic",
                accuracy=accuracy_score,
                precision=precision,
                recall=recall,
                f1=f1
            )
            
            print(synt_result)
            results.append(synt_result)

            # Gera o dado semi-sintético
            data_semi_synt = generate_semi_syntetic_dataset(original_dataset.dataset, gen,  {0: 0, 1: 0.15, 2: 0.15, 3: 0.15, 4: 0.15})
            # Treina e avalia o modelo
            trained_model = train_function(data_semi_synt.dataset.X_train_set, data_semi_synt.dataset.Y_train_set, clf.copy())
            accuracy_score, precision, recall, f1 = eval_function(original_dataset.dataset.X_test_set, original_dataset.dataset.Y_test_set, trained_model)
            semi_synt_result = ExperimentResult(
                generator_name=gen.get_name,
                classifier_name=clf.get_name(),
                data_type="semi-synthetic",
                accuracy=accuracy_score,
                precision=precision,
                recall=recall,
                f1=f1
            )
            print(semi_synt_result)
            results.append(semi_synt_result)

            # Gera o resultado com os valores originals
            trained_model = train_function(original_dataset.dataset.X_train_set, original_dataset.dataset.Y_train_set, clf.copy())
            accuracy_score, precision, recall, f1 = eval_function(original_dataset.dataset.X_test_set, original_dataset.dataset.Y_test_set, trained_model)
            original_result = ExperimentResult(
                generator_name=gen.get_name,
                classifier_name=clf.get_name(),
                data_type="original",
                accuracy=accuracy_score,
                precision=precision,
                recall=recall,
                f1=f1
            )
            print(original_result)
            results.append(original_result)


    # Salva os resultados em um arquivo csv
    results_df = pd.DataFrame([vars(result) for result in results])
    results_df.to_csv(os.path.join(save_path, "experiment_results.csv"), index=False)

def main():
    

    tts_cgan_model_path = "TTSCGAN/logs/TTS_APT_CGAN_OITO_VAR_IMPR7/Model/checkpoint"
    rcgan_model_path = "RGAN/experiments/settings/dapt2020.txt"
    
    RCGAN_generator = RCGANGEN.SyntGenerator(model_path=rcgan_model_path, epoch=89)
    TTSCGAN_generator = TTSCGANGenerator(seq_len=30, num_channels=8, num_classes=5, model_path=tts_cgan_model_path)

    generators = [TTSCGAN_generator, RCGAN_generator]

    classifiers = [RandomForestClassifierModel(50),
                   SVMClassifier(),
                   LSTMClassifier(8, 30, 64, 5),
                   TransformerClassifier(8, 30, 5)]

    # Carrega o dataset original
    original_dataset = load_original_dataset(is_train=True, attack_only=False, Shuffle=True)


    #data_semi_synt = generate_semi_syntetic_dataset(original_dataset.dataset, TTSCGAN_generator,  {0: 0, 1: 0.15, 2: 0.15, 3: 0.15, 4: 0.15})
    #trained_model = train_cpu_model(data_semi_synt.dataset.X_train_set, data_semi_synt.dataset.Y_train_set, RandomForestClassifierModel(50))
    #accuracy_score, precision, recall, f1 = evaluate_cpu_model(original_dataset.dataset.X_test_set, original_dataset.dataset.Y_test_set, trained_model)
    #print(f1)
    #
    #return

    # Roda os experimentos
    experiments_battery(generators, classifiers, original_dataset, save_path="experiments/results")

if __name__ == "__main__":
    #set 'TF_ENABLE_ONEDNN_OPTS' to '0' to avoid issues with TensorFlow
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    main()