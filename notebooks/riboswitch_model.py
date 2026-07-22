# -*- coding: utf-8 -*
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset, Subset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
from moe import MoE
from sklearn.metrics import f1_score, precision_score, recall_score
import time
import os
from sklearn.model_selection import KFold



parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--save_model", type=str, default="ribomoenet.pt")
parser.add_argument("--load_model", type=str, default=None)
args = parser.parse_args()

input_path = args.input

# Load processed dataset   
df = pd.read_csv(input_path).dropna(subset=["label"])

# test
print("Label value counts:")
print(df['label'].value_counts(dropna=False))
print("Unique label values:", df['label'].unique())

# Count the data
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
label_counts = df['label'].value_counts()
print(label_counts)

# Encode sequential data and pad with N
def one_hot_encode(seq, pad_char="N"):
    mapping = base_to_vec = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        'U': [0, 0, 0, 1],
        'N': [0, 0, 0, 0]
    }
    one_hot_encoded = []
    for sequence in seq:
        seqs = str(sequence)
        padded_seq = seqs.ljust(210, pad_char)
        encoded_seq = [base_to_vec.get(base, [0, 0, 0, 0]) for base in padded_seq]
        one_hot_encoded.append(encoded_seq)
    return np.array(one_hot_encoded)

encoded_seqs = one_hot_encode(df["complex_nucelotides"])
encoded_seqs = torch.tensor(encoded_seqs, dtype=torch.float32)
encoded_seqs_flat = encoded_seqs.flatten(start_dim=1)


encoded_seq_df = pd.DataFrame(encoded_seqs_flat, columns=[f"base_{i}" for i in range(encoded_seqs_flat.shape[1])], index=df.index)


# Select features and target
numerical_features = [
    "salt_molarity", "temperature (Celsius)", "trigger_target_binding_percentage", "switch_energy", "target_energy", 
    "reporter_energy", "complex_energy", "switch_gc", "switch_at", "target_gc", 
    "target_at", "reporter_gc", "reporter_at", "complex_gc", "complex_at",
    "switch_ensemble_defect", "target_ensemble_defect", "reporter_ensemble_defect",
    "complex_ensemble_defect", "switch_single_strandedness", "target_single_strandedness", 
    "reporter_single_strandedness", "complex_single_strandedness", "trigger_single_strandedness", "rbs_location", "loop_energy"
]

df[numerical_features] = df[numerical_features].fillna(0)
X = pd.concat([df[numerical_features], encoded_seq_df], axis=1)
y = df['label'] # index the label in real data


# Check for all-null rows
all_null_rows = df[df.isnull().all(axis=1)]
if not all_null_rows.empty:
    print("Rows with all null values found:")
    print(all_null_rows)
else:
    print("No rows with all null values.")

# Standardize features
scaler = StandardScaler()
X = torch.tensor(scaler.fit_transform(X), dtype=torch.float32)
y = torch.tensor(y.values, dtype=torch.long)


# First split: train+val and test
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Second split: 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
dataset_trainval = TensorDataset(X_trainval, y_trainval)


# Arguments
input_size = X.shape[1]  # Number of features
num_classes = 2  # Binary classification (True/False)
num_experts = 10
hidden_size = 64
k = 4
batch_size = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Training function
def train(dataloader, model, loss_fn, optim):
    model.train()
    loss = 0.0
    aux_loss = 0.0
    correct = 0
    total = 0
    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        y_hat, batch_aux_loss = model(x_batch)

        total_loss = loss_fn(y_hat, y_batch) + batch_aux_loss
        optim.zero_grad()
        total_loss.backward()
        optim.step()

        loss += total_loss.item()
        aux_loss += batch_aux_loss.item()
        # Accuracy
        preds = torch.argmax(y_hat, dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
    avg_loss = loss / len(dataloader)
    avg_aux_loss = aux_loss / len(dataloader)
    accuracy = correct / total * 100
    return avg_loss, avg_aux_loss, accuracy
    
# Loss defination   
loss_fn = nn.CrossEntropyLoss()

# Evaluation function
def eval(dataloader, model, loss_fn):
    model.eval()
    loss = 0.0
    aux_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_hat, batch_aux_loss = model(x_batch)
            total_loss = loss_fn(y_hat, y_batch) + batch_aux_loss
            loss += total_loss.item()
            aux_loss += batch_aux_loss.item()

            # Predictions
            probs = torch.softmax(y_hat, dim=1)
            
            # Accuracy
            preds = torch.argmax(y_hat, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

            # Tracking for metrics
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())


    avg_loss = loss / len(dataloader)
    avg_aux_loss = aux_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds)
    accuracy = correct / total * 100
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    return avg_loss, avg_aux_loss, f1, all_labels, all_preds, all_probs, accuracy, precision, recall



# 5-fold CV
epochs = 100

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset_trainval)):    
    print(f"\n-- Fold {fold+1} --")
    
    # Mini-batch training setup
    train_loader = DataLoader(Subset(dataset_trainval, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset_trainval, val_idx), batch_size=batch_size, shuffle=False)
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    val_aux_losses = []
    val_precision = []
    val_recall = []

    # Instantiate the MoE model
    model = MoE(input_size, num_classes, num_experts, hidden_size, k=k, noisy_gating=True).to(device)
    optim = Adam(model.parameters(), lr = 1e-4)

    for epoch in range(epochs):
        loss, aux_loss, accu = train(train_loader, model, loss_fn, optim)
        val_loss, val_aux_loss, _, _, _, _, val_acc, val_prec, val_rec = eval(val_loader, model, loss_fn)
        
        print(f"Epoch {epoch+1}: Train loss = {loss:.10f}, Train Aux loss = {aux_loss:.10f}, Accuracy = {accu:.10f}")
        print(f"Epoch {epoch+1}: Val Loss: {val_loss:.10f}, Val Aux Loss: {val_aux_loss:.10f}, Val Accuracy: {val_acc:.10f}, Val Precision:{val_prec:.10f}, Val Recall:{val_rec:.10f}")
        
        train_losses.append(loss)
        train_accuracies.append(accu)   

        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_aux_losses.append(val_aux_loss)
        val_precision.append(val_prec)
        val_recall.append(val_rec)    

    
    np.save(f"train_losses_8_fold{fold+1}.npy", np.array(train_losses))
    np.save(f"train_accuracies_8_fold{fold+1}.npy", np.array(train_accuracies))
    np.save(f"val_loss_8_fold{fold+1}.npy", np.array(val_losses))
    np.save(f"val_acc_8_fold{fold+1}.npy", np.array(val_accuracies))
    np.save(f"val_aux_losses_8_fold{fold+1}.npy", np.array(val_aux_losses))
    np.save(f"val_precision_8_fold{fold+1}.npy", np.array(val_precision))
    np.save(f"val_recall_8_fold{fold+1}.npy", np.array(val_recall))


# Train on full trainval dataset
final_model = MoE(input_size, num_classes, num_experts, hidden_size, k=k, noisy_gating=True).to(device)
optim = Adam(final_model.parameters(), lr = 1e-4) 

final_train_loader = DataLoader(dataset_trainval, batch_size=batch_size, shuffle=True)
final_train_losses = []
final_train_aux_losses = []
final_train_accuracies = []
print("\n===== Retraining Final Model on Full Train+Val Set =====")
train_start_time = time.time()
for epoch in range(epochs):
    loss, aux_loss, accu = train(final_train_loader, final_model, loss_fn, optim)
    final_train_losses.append(loss)
    final_train_aux_losses.append(aux_loss)
    final_train_accuracies.append(accu)   

    print(f"Epoch {epoch+1}: Train loss = {loss:.10f}, Train Aux loss = {aux_loss:.10f}, Accuracy = {accu:.10f}")

train_end_time = time.time()
print(f"Training Runtime: {train_end_time - train_start_time:.2f} seconds")
np.save(f"train_losses_8_final.npy", np.array(final_train_losses))
np.save(f"train_accuracies_8_final.npy", np.array(final_train_accuracies))
np.save(f"train_aux_losses_8_final.npy", np.array(final_train_aux_losses))
torch.save(final_model.state_dict(),"ribomoenet_final.pt")


# Mini-batch testing
final_model.load_state_dict(torch.load("ribomoenet_final.pt"))
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Evaluate on test set
test_start_time = time.time()
test_loss, test_aux_loss, test_f1, y_true, y_pred, y_probs, test_acc, test_prec, test_recall = eval(test_loader, final_model, loss_fn)
test_end_time = time.time()
print(f"Test Loss: {test_loss:.10f}, Test Aux Loss: {test_aux_loss:.10f}, F1: {test_f1:.10f}, Test Accuracy: {test_acc}, Test Precision: {test_prec}, Test recall: {test_recall}")
print(f"Evaluation Runtime: {test_end_time - test_start_time:.2f} seconds")
np.save("y_true_8.npy", np.array(y_true))
np.save("y_pred_8.npy", np.array(y_pred))
np.save("y_probs_8.npy", np.array(y_probs))
np.save("test_metrics_8.npy", np.array([test_loss, test_aux_loss, test_acc, test_f1, test_prec, test_recall]))








    

