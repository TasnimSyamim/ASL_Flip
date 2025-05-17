import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
import matplotlib.pyplot as plt

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, batch_first=False):
        super().__init__(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                         dropout=dropout, batch_first=batch_first)
        self.latest_attn_weights = None

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        # Pass is_causal to self.self_attn
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
            is_causal=is_causal  # Add this parameter
        )
        self.latest_attn_weights = attn_weights.detach()
        
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class CustomTransformer(nn.Module):
    def __init__(self, input_size, num_classes, d_model=64, nhead=8, num_encoder_layers=3, dim_feedforward=128, dropout=0.1):
        super(CustomTransformer, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 5000, d_model))
        self.encoder = nn.TransformerEncoder(
            CustomTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_encoder_layers,
        )
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_projection(x)
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :]
        x = self.encoder(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x
    
    
# Argument parser
parser = argparse.ArgumentParser(description="Train a Transformer model on WLASL dataset.")
parser.add_argument('--x_path', type=str, required=True, help="Path to the X dataset (numpy file).")
parser.add_argument('--y_path', type=str, required=True, help="Path to the y dataset (numpy file).")
parser.add_argument('--batch_size', type=int, default=32, help="Batch size for DataLoader.")
parser.add_argument('--hidden_size', type=int, default=64, help="Hidden size (d_model) of the Transformer.")
parser.add_argument('--num_layers', type=int, default=3, help="Number of layers in the Transformer encoder.")
parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for the optimizer.")
args = parser.parse_args()

# Load data
X = np.load(args.x_path)
y = np.load(args.y_path)

print(X.shape, y.shape)

y = tf.keras.utils.to_categorical(y, num_classes=100)

y_labels = np.argmax(y, axis=1)

X_train_ori, X_test_ori, y_train_ori, y_test_ori = train_test_split(X, y, test_size=0.2, stratify=y_labels, random_state=42)
X_test_ori, X_val_ori, y_test_ori, y_val_ori = train_test_split(X_test_ori, y_test_ori, test_size=0.5, stratify=y_test_ori.argmax(axis=1), random_state=42)
y_train_ori.shape, y_test_ori.shape, y_val_ori.shape

# Convert data to tensors
X_train = torch.tensor(X_train_ori, dtype=torch.float32)
X_test = torch.tensor(X_test_ori, dtype=torch.float32)
y_train = torch.tensor(y_train_ori.argmax(axis=1), dtype=torch.long)
y_test = torch.tensor(y_test_ori.argmax(axis=1), dtype=torch.long)
X_val = torch.tensor(X_val_ori, dtype=torch.float32)
y_val = torch.tensor(y_val_ori.argmax(axis=1), dtype=torch.long)

# DataLoader
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model, loss, and optimizer
input_size = X_train.size(-1)
num_classes = 100
model = CustomTransformer(input_size=input_size, num_classes=num_classes, d_model=args.hidden_size, num_encoder_layers=args.num_layers).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

# Create directory for saving attention weights
os.makedirs('attention_heatmaps', exist_ok=True)

# Training loop
num_epochs = 250
loss_history = []
val_loss_history = []

# Loss threshold
loss_threshold = 0.1

# Early stopping parameters
patience = 10  # Number of epochs to wait for improvement
best_val_loss = float('inf')
best_val_accuracy = 0.0
no_improvement_epochs = 0

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    
    for X_batch, y_batch in train_loader:
        # Move batch to GPU
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)  # Forward pass
        loss = criterion(outputs, y_batch)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    loss_history.append(avg_loss)
    
    # Validation phase
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (X_batch, y_batch) in enumerate(val_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)  # Forward pass
            loss = criterion(outputs, y_batch)  # Compute loss
            val_loss += loss.item()  # Accumulate validation loss

            # Collect attention weights for the first batch
            if batch_idx == 0:
                attention_weights = [layer.latest_attn_weights for layer in model.encoder.layers]

            # Compute accuracy
            predictions = outputs.argmax(dim=1)
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_loss_history.append(avg_val_loss)
    val_accuracy = correct / total

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    # Save attention weights every 10 epochs
    if (epoch + 1) % 10 == 0:
        attention_weights_cpu = [attn.cpu() for attn in attention_weights]
        torch.save(attention_weights_cpu, f'attention_heatmaps/attn_weights_epoch{epoch+1}.pt')

    # Check for improvement
    if avg_val_loss < best_val_loss or val_accuracy > best_val_accuracy:
        best_val_loss = min(avg_val_loss, best_val_loss)
        best_val_accuracy = max(val_accuracy, best_val_accuracy)
        no_improvement_epochs = 0  # Reset counter
    else:
        no_improvement_epochs += 1

    # Early stopping condition
    if no_improvement_epochs >= patience:
        print(f'Early stopping triggered after {patience} epochs with no improvement.')
        break

    # Loss threshold condition
    if avg_loss < loss_threshold:
        print(f'Loss threshold of {loss_threshold} reached. Stopping training.')
        break

# Evaluate the model
model.eval()
with torch.no_grad():
    # Move test data to GPU
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    accuracy = (test_outputs.argmax(dim=1) == y_test).float().mean()
    print(f'Test Loss: {test_loss.item():.4f}, Test Accuracy: {accuracy.item():.4f}')