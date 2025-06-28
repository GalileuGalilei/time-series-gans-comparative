from data_utils import *
from classifiers.Classifiers import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class PerClassEvaluation:
    def __init__(self, class_name, precision, recall, f1, support, false_positives, false_negatives):
        self.class_id = class_name
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.support = support
        self.false_positives = false_positives
        self.false_negatives = false_negatives

    def to_csv_row(self):
        return f"{self.class_id},{self.precision:.4f},{self.recall:.4f},{self.f1:.4f},{self.support},{self.false_positives},{self.false_negatives}"


class EvaluationReport:
    def __init__(self, classifier, data_type):
        self.per_class_evaluations = []
        self.data_type = data_type  # 'original', 'synthetic', or 'semi-synthetic'
        self.classifier = classifier  # Name of the classifier used for this report

    def add_class_evaluation(self, class_evaluation: PerClassEvaluation):
        self.per_class_evaluations.append(class_evaluation)

    def calculate_overall_metrics(self):
        if not self.per_class_evaluations:
            return 0.0, 0.0, 0.0

        total_tp = sum(e.recall * e.support for e in self.per_class_evaluations)
        total_fp = sum(e.false_positives for e in self.per_class_evaluations)
        total_fn = sum(e.false_negatives for e in self.per_class_evaluations)

        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = (2 * overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0

        return overall_precision, overall_recall, overall_f1

    def to_csv_header():
        return "Classifier,Data Type,Class Name,Precision,Recall,F1 Score,Support,False Positives,False Negatives"

    def __str__(self):
        report_lines = []
        for evaluation in self.per_class_evaluations:
            line = evaluation.to_csv_row()
            line = f"{self.classifier},{self.data_type},{line}"
            report_lines.append(line)

        overall_precision, overall_recall, overall_f1 = self.calculate_overall_metrics()
        report_lines.append(f"{self.classifier},{self.data_type},Overall,{overall_precision:.4f},{overall_recall:.4f},{overall_f1:.4f},-,-,-")
        
        return "\n".join(report_lines)

def train_cpu_model(X_set, Y_set, model):
    print("Training Classifier")

    model.fit(X_set, Y_set)
    print("Training completed")
    return model

def evaluate_cpu_model(X_set, Y_set, model, classes_by_id=None, data_type="unknown"):
    prediction = model.predict(X_set)
    all_preds = np.array(prediction)
    all_labels = np.array(Y_set)

    # Matriz de Confusão
    conf_matrix = confusion_matrix(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_ids = np.unique(all_labels)
    eval_report = EvaluationReport(classifier=model.get_name(), data_type=data_type)

    for class_id in class_ids:
        tp = conf_matrix[class_id, class_id]
        fp = conf_matrix[:, class_id].sum() - tp
        fn = conf_matrix[class_id, :].sum() - tp
        support = conf_matrix[class_id, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        eval_report.add_class_evaluation(PerClassEvaluation(
            class_name = classes_by_id[class_id] if classes_by_id is not None else str(class_id),
            precision=precision,
            recall=recall,
            f1=f1,
            support=int(support),
            false_positives=int(fp),
            false_negatives=int(fn)
        ))

    return eval_report

    

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

def evaluate_torch_model(X_test_set, Y_test_set, model, classes_by_id=None, data_type="unknown"):
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

    # Matriz de Confusão
    conf_matrix = confusion_matrix(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_ids = np.unique(all_labels)
    eval_report = EvaluationReport(classifier=model.get_name(), data_type=data_type)

    for class_id in class_ids:
        tp = conf_matrix[class_id, class_id]
        fp = conf_matrix[:, class_id].sum() - tp
        fn = conf_matrix[class_id, :].sum() - tp
        support = conf_matrix[class_id, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        eval_report.add_class_evaluation(PerClassEvaluation(
            class_name = classes_by_id[class_id] if classes_by_id is not None else str(class_id),
            precision=precision,
            recall=recall,
            f1=f1,
            support=int(support),
            false_positives=int(fp),
            false_negatives=int(fn)
        ))

    return eval_report

def experiments_battery(generators : list[IGenerator], classifiers : list[IClassifier], original_dataset : DataLoader, save_path: str = "experiments/"):
    """
    Roda uma bateria de experimentos com diferentes geradores e salva os resultados.
    """

    #verifica se o caminho de salvamento existe, se não, cria
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    #todo: corrigir essa nomeclatura
    classes_by_id = original_dataset.dataset.classes

    for gen in generators:
        results = []
        for clf in classifiers:
            print(f"Running experiment with generator: {gen.get_name} and classifier: {clf.get_name}")

            # Como alguns modelos só funcionam em CPU ou GPU, escolhe a função de treinamento e avaliação apropriada
            (train_function, eval_function) = (train_torch_model, evaluate_torch_model) if clf.is_torch_model() else (train_cpu_model, evaluate_cpu_model)

            # sintético
            data_synt = generate_syntetic_dataset(original_dataset.dataset, gen)
            trained_model = train_function(data_synt.dataset.X_train_set, data_synt.dataset.Y_train_set, clf.copy())
            report = eval_function(original_dataset.dataset.X_test_set, original_dataset.dataset.Y_test_set, trained_model, classes_by_id, data_type='synthetic')
            results.append(report)

            print(report)

            # semi-sintético
            data_semi_synt = generate_semi_syntetic_dataset(original_dataset.dataset, gen,  {0: 0, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2})
            trained_model = train_function(data_semi_synt.dataset.X_train_set, data_semi_synt.dataset.Y_train_set, clf.copy())
            report = eval_function(original_dataset.dataset.X_test_set, original_dataset.dataset.Y_test_set, trained_model, classes_by_id, data_type='semi-synthetic')
            results.append(report)

            print(report)

            # originais
            trained_model = train_function(original_dataset.dataset.X_train_set, original_dataset.dataset.Y_train_set, clf.copy())
            report = eval_function(original_dataset.dataset.X_test_set, original_dataset.dataset.Y_test_set, trained_model, classes_by_id, data_type='original')
            results.append(report)

            print(report)

        #um único csv por gerador
        with open(os.path.join(save_path, f"{gen.get_name}_results.csv"), "w") as f:
            f.write(EvaluationReport.to_csv_header() + "\n")
            for report in results:
                f.write(str(report) + "\n")


def main():
    
    tts_cgan_model_path = "logs/TTS_APT_CGAN_16_VAR_2025_06_27_17_21_28/Model/checkpoint"
    rcgan_model_path = "RGAN/experiments/settings/dapt2020.txt"
    time_gan_model_path = "output/TimeGAN/dapt_v2/train/weights"

    # Carrega os datasets originais
    original_dataset = load_original_dataset(is_train=True, attack_only=False, Shuffle=True)
    
    # Cria os geradores
    generators = [#RCGAN.SyntheticGenerator(model_path=rcgan_model_path, epoch=89),
                  TTSCGAN.SyntheticGenerator(seq_len=30, num_channels=6, num_classes=5, model_path=tts_cgan_model_path)]
                  #TimeGAN.SyntheticGenerator(model_path=time_gan_model_path, data=original_dataset.dataset)]

    # Cria os classificadores
    classifiers = [RandomForestClassifierModel(50),
                   SVMClassifier(),
                   LSTMClassifier(6, 30, 64, 5),
                   TransformerClassifier(6, 30, 5)]

    # Roda os experimentos
    experiments_battery(generators, classifiers, original_dataset, save_path="experiments/results")

if __name__ == "__main__":
    #set 'TF_ENABLE_ONEDNN_OPTS' to '0' to avoid issues with TensorFlow
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    main()