import torch
print("HELLO")
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse

# Set matplotlib backend to 'Agg' to avoid display issues (useful for non-GUI environments)
import matplotlib
matplotlib.use('Agg')

# Directory where attention weights are saved
attn_dir = 'attention_heatmaps'

parser = argparse.ArgumentParser(description="Visualize Attention Heatmap.")
parser.add_argument('--num_layers', type=int, default=3, help="Number of layers in the Transformer encoder.")
args = parser.parse_args()

# Number of epochs and layers (adjust based on your training)
num_epochs = 250  # Match your training loop
num_encoder_layers = args.num_layers  # From your model definition

# Create a subdirectory for heatmap images
os.makedirs(os.path.join(attn_dir, 'images'), exist_ok=True)

# Visualize attention for every 10 epochs
for epoch in range(10, num_epochs + 1, 10):
    attn_weights_path = os.path.join(attn_dir, f'attn_weights_epoch{epoch}.pt')
    if os.path.exists(attn_weights_path):
        attention_weights = torch.load(attn_weights_path)
        sample_idx = 0  # Visualize the first sample in the batch

        for layer_idx in range(len(attention_weights)):
            # Average attention weights over all heads
            attn_matrix_avg = attention_weights[layer_idx][sample_idx].mean(dim=0).numpy()
            
            # Create and save the heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(attn_matrix_avg, cmap='viridis', square=True)
            plt.title(f'Epoch {epoch}, Layer {layer_idx + 1}, Average over Heads')
            plt.xlabel('Source Token Position')
            plt.ylabel('Target Token Position')
            plt.savefig(os.path.join(attn_dir, 'images', f'heatmap_epoch{epoch}_layer{layer_idx + 1}_avg.png'))
            plt.close()