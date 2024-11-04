import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
import random
import os

NUM_CLIENTS = 5
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-4  # Ajustado conforme sugestão
WEIGHT_DECAY = 1e-4  # Aumentado para regularização
FRACTION = 1.0  
PATIENCE = 5  # Para Early Stopping

def set_seed(seed=42):
    """Define a semente para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob=0.5):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),  # Reduzido de hidden_dim * 2 para hidden_dim
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

def create_clients(X, y, num_clients=5, initial='client'):
    """Distribui os dados de forma aleatória entre os clientes.
    
    Args:
        X (np.ndarray): Features.
        y (np.ndarray): Labels.
        num_clients (int): Número de clientes.
        initial (str): Prefixo para os nomes dos clientes.
    
    Returns:
        dict: Dicionário com nomes dos clientes e seus respectivos shards de dados.
    """
    client_names = [f'{initial}_{i+1}' for i in range(num_clients)]
    
    # Embaralhar os dados
    data = list(zip(X, y))
    random.shuffle(data)
    
    # Dividir os dados em shards
    size = len(data) // num_clients
    shards = [data[i:i + size] for i in range(0, size * num_clients, size)]
    
    # Assegurar que o número de shards corresponde ao número de clientes
    assert len(shards) == len(client_names), "Número de shards não corresponde ao número de clientes."
    
    return {client_names[i]: shards[i] for i in range(len(client_names))}

def create_client_loaders(clients, batch_size=32):
    """Cria DataLoaders para cada cliente.
    
    Args:
        clients (dict): Dicionário com shards de dados por cliente.
        batch_size (int): Tamanho do batch.
    
    Returns:
        dict: Dicionário com DataLoaders para cada cliente.
    """
    client_loaders = {}
    for client_name, data_shard in clients.items():
        X_client, y_client = zip(*data_shard)
        X_client = torch.tensor(X_client, dtype=torch.float32)
        y_client = torch.tensor(y_client, dtype=torch.long)
        dataset = TensorDataset(X_client, y_client)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        client_loaders[client_name] = loader
    return client_loaders

def train_client(model, loader, criterion, optimizer, device):
    """Treina o modelo em um cliente e retorna a perda média."""
    model.train()
    total_loss = 0.0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    """Avalia o modelo no conjunto de teste.
    
    Args:
        model (nn.Module): Modelo a ser avaliado.
        loader (DataLoader): DataLoader de teste.
        device: Dispositivo (CPU/GPU).
    
    Returns:
        tuple: Acurácia, precisão, recall e F1-score.
    """
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.numpy())
    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
    return accuracy, precision, recall, f1

def evaluate_full_metrics(model, loader, device):
    """Avalia métricas adicionais do modelo."""
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            outputs = model(data)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(target.numpy())
    
    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
    cm = confusion_matrix(all_targets, all_preds)
    try:
        auc = roc_auc_score(all_targets, all_probs, multi_class='ovo')
    except ValueError:
        auc = "AUC não aplicável para multiclasse sem probabilidade adequada."
    
    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Matriz de Confusão:")
    print(cm)
    print(f"AUC: {auc}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv("/content/drive/MyDrive/dataset_with_hc_tratado.csv")

    X = df.drop('target', axis=1).values
    y = df['target'].values

    # Divisão dos dados em treino, validação e teste
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1, random_state=42, stratify=y_train_val
    )

    # Normalização dos dados
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Criação dos clientes com shards de dados
    clients = create_clients(X_train, y_train, num_clients=NUM_CLIENTS, initial='client')
    client_loaders = create_client_loaders(clients, batch_size=BATCH_SIZE)

    # Preparação dos DataLoaders de validação e teste
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = X.shape[1]
    hidden_dim = 128
    output_dim = len(np.unique(y))
    print(f"Input dim: {input_dim}")
    print(f"Output dim: {output_dim}")

    # Inicialização do modelo global
    global_model = MLP(input_dim, hidden_dim, output_dim, dropout_prob=0.5).to(device)
    global_model.train()

    # Cálculo de pesos das classes para balanceamento
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_loss = float('inf')
    trigger_times = 0

    for epoch in range(EPOCHS):
        local_weights = []
        epoch_loss = 0.0
        print(f"Rodada {epoch+1}/{EPOCHS}")
        for client_idx, (client_name, loader) in enumerate(client_loaders.items()):
            local_model = MLP(input_dim, hidden_dim, output_dim, dropout_prob=0.5).to(device)
            local_model.load_state_dict(global_model.state_dict())
            optimizer = optim.Adam(local_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            
            loss = train_client(local_model, loader, criterion, optimizer, device)
            epoch_loss += loss
            local_weights.append(local_model.state_dict())
            print(f"  Cliente {client_idx+1} treinado. Loss: {loss:.4f}")
        
        epoch_loss /= NUM_CLIENTS
        print(f"  Loss média da rodada: {epoch_loss:.4f}")

        # Agregação dos pesos dos clientes
        global_dict = global_model.state_dict()
        for key in global_dict.keys():
            # Média dos pesos de todos os clientes
            global_dict[key] = torch.stack([local_weights[i][key].float() for i in range(NUM_CLIENTS)], 0).mean(0)
        global_model.load_state_dict(global_dict)
        print("  Pesos agregados.")

        # Avaliação no conjunto de validação
        global_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = global_model(data)
                loss = criterion(outputs, target)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"  Val Loss: {val_loss:.4f}")

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            # Salvar o melhor modelo
            torch.save(global_model.state_dict(), "best_model.pth")
            print("  Melhor modelo salvo.")
        else:
            trigger_times += 1
            print(f"  Trigger times: {trigger_times}")
            if trigger_times >= PATIENCE:
                print("  Early stopping acionado.")
                break
        print()

    # Carregar o melhor modelo
    if os.path.exists("best_model.pth"):
        global_model.load_state_dict(torch.load("best_model.pth"))
        print("Melhor modelo carregado para avaliação final.\n")

    # Avaliação final do modelo global
    accuracy, precision, recall, f1 = evaluate(global_model, test_loader, device)
    print("Métricas de Avaliação no Conjunto de Teste:")
    print(f"  Acurácia: {accuracy:.4f}")
    print(f"  Precisão: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print("\nRelatório de Classificação:")
    y_pred = []
    y_true = []
    global_model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            outputs = global_model(data)
            _, preds = torch.max(outputs, 1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(target.numpy())
    print(classification_report(y_true, y_pred))
    
    # Avaliação com métricas adicionais
    from sklearn.metrics import confusion_matrix, roc_auc_score
    try:
        evaluate_full_metrics(global_model, test_loader, device)
    except Exception as e:
        print(f"Erro ao calcular métricas adicionais: {e}")

if __name__ == "__main__":
    main()