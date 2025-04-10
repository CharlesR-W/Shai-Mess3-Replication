import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import random
import sys

%matplotlib inline

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Configuration variables - modify these to change behavior
GENERATE_CONTROL_PLOT = True  # Set to True to generate control plot of all possible sequences
NUM_EPOCHS = 100               # Number of epochs to train
BATCH_SIZE = 128               # Batch size for training
SEQ_LENGTH = 10                # Length of sequences to generate
CONTEXT_WINDOW = 10            # Context window size for transformer
LEARNING_RATE = 0.01           # Learning rate
SAVE_INTERVAL = 10           # Interval for saving checkpoints (None = auto-calculate)
PLOT_INTERVAL = None           # Interval for plotting belief states (None = auto-calculate)
CONTROL_PLOT_RANDOM = True     # Whether to use random sampling for control plot (None = auto-determine based on seq_length)
HMM_MODEL_TYPE = "z1r"       # Type of HMM model to use: "mess3" or "z1r"

class HMMModel:
    def __init__(self, x=0.05, alpha=0.85, device='cpu', model_type="mess3"):
        self.x = x
        self.alpha = alpha
        self.beta = (1 - alpha) / 2.0
        self.y = 1 - 2 * x
        self.dim = 3
        self.emit_values = [0, 1, 2]
        self.states_values = [0, 1, 2]
        self.draw_values = list(product(self.emit_values, self.states_values))
        self.device = device
        self.model_type = model_type
        
        # Create transition matrix based on model type
        if model_type == "mess3":
            # Original MESS3 model
            self.transition = torch.zeros((self.dim, self.dim, self.dim), device=device)
            for lvi, lvj, lvk in product(range(self.dim), repeat=3):
                self.transition[lvi, lvj, lvk] = (self.y if lvi == lvj else self.x) * (self.alpha if lvj == lvk else self.beta)
        elif model_type == "z1r":
            # Z1R model:
            # State 0: always emits 0, always transitions to state 1
            # State 1: always emits 1, always transitions to state R
            # State R: emits 0 or 1 with equal probability, always transitions to state 0
            self.transition = torch.zeros((self.dim, self.dim, self.dim), device=device)
            
            # State 0: always emits 0, always transitions to state 1
            self.transition[0, 1, 0] = 1.0
            
            # State 1: always emits 1, always transitions to state R (state 2)
            self.transition[1, 2, 1] = 1.0
            
            # State R (state 2): emits 0 or 1 with equal probability, always transitions to state 0
            self.transition[2, 0, 0] = 0.5  # Emit 0, transition to state 0
            self.transition[2, 0, 1] = 0.5  # Emit 1, transition to state 0
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Calculate stationary distribution
        self.stationary_dist = self._calculate_stationary_distribution()
    
    def _calculate_stationary_distribution(self):
        """Calculate the stationary distribution of the HMM"""
        # Create the transition matrix for the Markov chain (marginalizing over emissions)
        P = self.transition.sum(dim=2)
        
        # Find the eigenvector corresponding to eigenvalue 1
        eigenvalues, eigenvectors = torch.linalg.eig(P.T)
        
        # Find the index of the eigenvalue closest to 1
        idx = torch.argmin(torch.abs(eigenvalues.real - 1.0))
        # Get the corresponding eigenvector
        stationary = eigenvectors[:, idx].real
        
        # Normalize to get a probability distribution
        stationary = stationary / stationary.sum()
        
        return stationary
    
    def generate_sequences(self, seq_length, batch_size):
        """
        Generate sequences from the HMM model:
        initial state is sampled according to stationary distribution
        returns shape (batch_size, seq_length)
        """        
        sequences = torch.zeros((batch_size, seq_length), dtype=torch.long, device=self.device)
        #states = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        states = torch.multinomial(self.stationary_dist, num_samples=batch_size, replacement=True)

        for t in range(seq_length):
            # For each sample in the batch
            for i in range(batch_size):
                # Get probabilities for current state
                probs = self.transition[states[i]]
                # Sample emission and next state
                idx = torch.multinomial(probs.view(-1), num_samples=1).item()
                next_state, emit = self.draw_values[idx]
                sequences[i, t] = emit
                states[i] = next_state
        
        return sequences
    
    def get_optimal_belief_states(self, sequences):
        """Get the optimal belief states for each sequence using Bayes' rule"""
        batch_size, seq_length = sequences.shape
        belief_states = torch.zeros((batch_size, seq_length, self.dim), device=sequences.device)
        
        # Initialize with stationary distribution
        for i in range(batch_size):
            belief_states[i, 0] = self.stationary_dist.to(sequences.device)
        
        # Update belief states using Bayes' rule
        for i in range(batch_size):
            for t in range(1, seq_length):
                # Get the observed symbol
                symbol = sequences[i, t-1]
                
                # Get the current belief state
                current_belief = belief_states[i, t-1]
                
                Tx = self.transition[:,:,symbol]
                updated_belief = current_belief @ Tx

                # Renormalize to get a valid probability distribution
                updated_belief = updated_belief / updated_belief.sum()
                
                # Store the updated belief state
                belief_states[i, t] = updated_belief
        
        return belief_states

class TransformerWithResidual(nn.Module):
    def __init__(self, vocab_size=3, d_model=64, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Add layer normalization before the transformer
        self.pre_norm = nn.LayerNorm(d_model)
        
        # Transformer encoder layers with residual connections
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Use pre-norm architecture (more stable)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Add layer normalization after the transformer
        self.post_norm = nn.LayerNorm(d_model)
        
        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Projection matrix for residual points
        self.projection_matrix = None
        self.projection_bias = None
        
        # Initialize parameters
        self.init_parameters()
    
    def init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, src_mask=None):
        # src shape: [batch_size, seq_len]
        src = self.embedding(src)
        src = self.pos_encoder(src)
        
        # Apply pre-norm layer normalization
        src = self.pre_norm(src)
        
        # Transformer encoder with residual connections and causal mask
        output = self.transformer_encoder(src, src_key_padding_mask=None, mask=src_mask)
        
        # Apply post-norm layer normalization
        output = self.post_norm(output)
        
        # Project to vocabulary size
        output = self.output_layer(output)
        
        return output
    
    def get_residuals(self, src, src_mask=None):
        """Get the residual activations before the final layer"""
        src = self.embedding(src)
        src = self.pos_encoder(src)
        
        # Apply pre-norm layer normalization
        src = self.pre_norm(src)
        
        # Get transformer output
        residuals = self.transformer_encoder(src, src_key_padding_mask=None, mask=src_mask)
        
        # Apply post-norm layer normalization
        residuals = self.post_norm(residuals)
        
        return residuals
    
    def optimize_projection_matrix(self, residuals, true_belief_states, num_states=3):
        """Find projection matrix W and bias c that minimize mean-square error between projected points and ground-truth optimal belief states"""
        # Convert to numpy for optimization
        residuals_np = residuals.detach().cpu().numpy()
        true_belief_np = true_belief_states.detach().cpu().numpy()
        
        # Reshape for optimization - flatten batch and time dimensions only
        residuals_flat = residuals_np.reshape(-1, residuals_np.shape[-1])  # Shape: (batch*time, d_model)
        true_belief_flat = true_belief_np.reshape(-1, true_belief_np.shape[-1])  # Shape: (batch*time, num_states)
        
        # Ensure both tensors have the same number of rows
        min_rows = min(residuals_flat.shape[0], true_belief_flat.shape[0])
        residuals_flat = residuals_flat[:min_rows]
        true_belief_flat = true_belief_flat[:min_rows]
        
        # Center and scale the residuals
        # 1. Center: subtract the mean vector
        residuals_mean = np.mean(residuals_flat, axis=0)
        residuals_centered = residuals_flat - residuals_mean
        
        # 2. Scale: divide by the standard deviation of each axis
        residuals_std = np.std(residuals_centered, axis=0)
        residuals_scaled = residuals_centered / residuals_std
        
        # Add a column of ones to the data for the bias term
        X = np.column_stack([residuals_scaled, np.ones(residuals_scaled.shape[0])])  # Shape: (batch*time, d_model+1)
        assert np.linalg.matrix_rank(X) == X.shape[1]
        # Use the true belief states as the target
        Y = true_belief_flat
        
        # Solve the least squares problem: min ||X[W;c] - Y||_F^2 directly
        # Stack X matrices side by side for all columns of Y
        X_stacked = np.kron(np.eye(Y.shape[1]), X)  # Shape: (batch*time*num_states, (d_model+1)*num_states)
        Y_stacked = Y.flatten()  # Shape: (batch*time*num_states,)
        
        # Solve the combined system
        W_c_stacked = np.linalg.lstsq(X_stacked, Y_stacked, rcond=1e-5)[0]
        
        # Reshape solution back to matrix form
        W_c = W_c_stacked.reshape(Y.shape[1], -1).T  # Shape: (d_model+1, num_states)
        
        # Verify rank
        rank = np.linalg.matrix_rank(W_c)
        assert rank >= 2  # Should have rank at least 2 since belief states sum to 1
        
        # Extract W and c
        W = W_c[:-1]  # Projection matrix
        c = W_c[-1]   # Bias vector
        
        # Store the projection matrix, bias, and scaling parameters
        self.projection_matrix = torch.tensor(W, dtype=torch.float32, device=residuals.device)
        self.projection_bias = torch.tensor(c, dtype=torch.float32, device=residuals.device)
        #self.residuals_mean = torch.tensor(residuals_mean, dtype=torch.float32, device=residuals.device)
        #self.residuals_std = torch.tensor(residuals_std, dtype=torch.float32, device=residuals.device)
        
        assert torch.linalg.matrix_rank(self.projection_matrix).item() > 1

        return self.projection_matrix, self.projection_bias
    
    def project_residuals(self, residuals):
        """Project residual points using the optimized projection matrix and bias"""
        if self.projection_matrix is None:
            raise ValueError("Projection matrix has not been optimized yet")
        
        # Apply the projection
        batch_size, seq_length, d_model = residuals.shape
        residuals_flat = residuals.reshape(-1, d_model)
        
        # Center and scale the residuals
        residuals_mean = torch.mean(residuals_flat, axis=0)
        residuals_std = torch.std(residuals_flat) #keep relative variances, just scale
        residuals_centered = residuals_flat - residuals_mean
        residuals_scaled = residuals_centered / residuals_std
        
        # Apply projection: y = W^T x + c
        projected = torch.mm(residuals_scaled, self.projection_matrix) + self.projection_bias
        
        projected_rescaled = (projected - torch.mean(projected, axis=0))/torch.std(projected)
        projected_rescaled = projected_rescaled.reshape(batch_size, seq_length, -1)
        return projected_rescaled

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def plot_belief_states(projected_residuals, true_belief_states, epoch, save_dir='plots', plot_2d=False):
    """Plot belief states by projecting onto the plane with normal (1,1,1) and showing the view normal to that plane"""
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to numpy for plotting
    projected_np = projected_residuals.detach().cpu().numpy()
    true_belief_np = true_belief_states.detach().cpu().numpy()
    
    # Reshape for plotting - each token is a point
    batch_size, seq_length, num_states = projected_np.shape
    projected_flat = projected_np.reshape(-1, num_states)
    true_belief_flat = true_belief_np.reshape(-1, num_states)
    

    
    # Color mixture of belief states
    colors = true_belief_flat

    
    
    if plot_2d:
        # Direct projection matrix to project onto the plane with normal (1,1,1)
        normal = np.array([1, 1, 1])
        normal_norm_squared = np.sum(normal**2)
        projection_matrix = np.eye(3) - np.outer(normal, normal) / normal_norm_squared
        # Apply the projection matrix to each point
        projected_triangle = np.dot(projected_flat, projection_matrix)

        # Convert from 3D to 2D coordinates
        projected_2d = projected_triangle[:, :2]
        
        # Create the plot
        plt.figure(figsize=(10, 10))
        
        # Define the vertices of the equilateral triangle
        vertices = np.array([[0, 0], [1, 0], [0, 1]])
        
        # Plot the triangle
        plt.plot([vertices[0, 0], vertices[1, 0], vertices[2, 0], vertices[0, 0]], 
                [vertices[0, 1], vertices[1, 1], vertices[2, 1], vertices[0, 1]], 'k-', linewidth=2)
        
        # Add vertex labels
        plt.text(vertices[0, 0] - 0.1, vertices[0, 1] - 0.1, 'State 0', fontsize=14, ha='right', va='top')
        plt.text(vertices[1, 0] + 0.1, vertices[1, 1] - 0.1, 'State 1', fontsize=14, ha='left', va='top')
        plt.text(vertices[2, 0], vertices[2, 1] + 0.1, 'State 2', fontsize=14, ha='center', va='bottom')
        
        # Plot all points with colors based on the true belief state
        scatter = plt.scatter(
            projected_2d[:, 0], 
            projected_2d[:, 1], 
            c=colors, 
            alpha=0.7,
            s=50,  # Increased point size
            edgecolors='black',  # Add black edges for better visibility
            linewidth=0.5
        )
        
        # Add title
        plt.title(f'Projected Belief States (Epoch {epoch+1})', pad=20, fontsize=16)
        
        # Set the plot limits
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        
        # Remove axis labels and ticks
        plt.axis('off')
        
        # Save the plot
        plt.savefig(os.path.join(save_dir, f'belief_states_epoch_{epoch+1}.png'), bbox_inches='tight', dpi=150)
    
        # Close the plot
        plt.close()
    
    # Create a 3D view of the belief states
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the 3D triangle
    triangle_3d = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    ax.plot_trisurf(triangle_3d[:, 0], triangle_3d[:, 1], triangle_3d[:, 2], 
                    color='lightgray', alpha=0.3)
    
    # Plot the projected points in 3D
    
    # Plot the points
    scatter = ax.scatter(
        projected_flat[:, 0], 
        projected_flat[:, 1], 
        projected_flat[:, 2], 
        c=colors, 
        alpha=0.7,
        s=50,  # Increased point size
        edgecolors='black',  # Add black edges for better visibility
        linewidth=0.5
    )
    
    # Set the view to be normal to the plane (1,1,1)
    ax.view_init(elev=35, azim=45)
    
    # Set the axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    
    # Add labels
    ax.set_xlabel('State 0', fontsize=12)
    ax.set_ylabel('State 1', fontsize=12)
    ax.set_zlabel('State 2', fontsize=12)
    # Add title
    ax.set_title(f'3D View of Belief States (Epoch {epoch+1})', pad=20, fontsize=16)
    
    # Save the 3D plot
    plt.savefig(os.path.join(save_dir, f'belief_states_3d_epoch_{epoch+1}.png'), bbox_inches='tight', dpi=150)
    
    # Close the plot
    plt.close()

def plot_all_possible_sequences(hmm_model, seq_length=10, use_random=None, save_dir='plots'):
    """Generate and plot all possible sequences on the belief triangle"""
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Process in batches to avoid memory issues
    batch_size = 1000
    
    # Determine whether to generate all possible sequences or sample randomly
    if use_random is None:
        use_random = seq_length > 10
    
    if not use_random:
        # Generate all possible sequences
        total_sequences = 3 ** seq_length
        print(f"Generating all {total_sequences} possible sequences of length {seq_length}")
        
        all_sequences = []
        all_belief_states = []
        
        # Generate all possible sequences systematically
        for i in range(0, total_sequences, batch_size):
            current_batch_size = min(batch_size, total_sequences - i)
            print(f"Processing batch {i//batch_size + 1}/{(total_sequences + batch_size - 1)//batch_size}")
            
            # Generate sequences for this batch
            sequences = torch.zeros((current_batch_size, seq_length), dtype=torch.long, device=hmm_model.device)
            
            # Fill sequences with values based on the batch index
            for j in range(current_batch_size):
                seq_idx = i + j
                # Convert to base-3 and pad with zeros
                for k in range(seq_length):
                    # Extract the k-th digit in base-3
                    digit = (seq_idx // (3 ** k)) % 3
                    sequences[j, seq_length - 1 - k] = digit
            
            # Get belief states for these sequences
            belief_states = hmm_model.get_optimal_belief_states(sequences)
            
            all_sequences.append(sequences)
            all_belief_states.append(belief_states)
        
        # Concatenate all batches
        sequences = torch.cat(all_sequences, dim=0)
        belief_states = torch.cat(all_belief_states, dim=0)
    else:
        # For longer sequences, sample randomly
        num_samples = 10000
        print(f"Sampling {num_samples} random sequences of length {seq_length}")
        
        all_sequences = []
        all_belief_states = []
        
        for i in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - i)
            print(f"Processing batch {i//batch_size + 1}/{(num_samples + batch_size - 1)//batch_size}")
            
            # Generate random sequences
            sequences = hmm_model.generate_sequences(seq_length, current_batch_size)
            
            # Get belief states for these sequences
            belief_states = hmm_model.get_optimal_belief_states(sequences)
            
            all_sequences.append(sequences)
            all_belief_states.append(belief_states)
        
        # Concatenate all batches
        sequences = torch.cat(all_sequences, dim=0)
        belief_states = torch.cat(all_belief_states, dim=0)
        total_sequences = num_samples
    
    # Convert to numpy for plotting
    belief_states_np = belief_states.detach().cpu().numpy()
    
    # Reshape for plotting
    batch_size, seq_length, num_states = belief_states_np.shape
    belief_states_flat = belief_states_np.reshape(-1, num_states)
    
    # Print statistics about the belief states
    print(f"Belief state statistics:")
    print(f"  Min values: {np.min(belief_states_flat, axis=0)}")
    print(f"  Max values: {np.max(belief_states_flat, axis=0)}")
    print(f"  Mean values: {np.mean(belief_states_flat, axis=0)}")
    print(f"  Sum of each belief state (should be close to 1): {np.sum(belief_states_flat, axis=1).mean():.4f}")
    
    # Convert to triangle coordinates
    belief_states_triangle = belief_states_flat[:, :2]  # drop z-coordinate
    
    # Create the 2D plot
    plt.figure(figsize=(10, 10))
    
    # Define the vertices of the equilateral triangle
    vertices = np.array([[0, 0], [1, 0], [0, 1]])
    
    # Plot the triangle
    plt.plot([vertices[0, 0], vertices[1, 0], vertices[2, 0], vertices[0, 0]], 
             [vertices[0, 1], vertices[1, 1], vertices[2, 1], vertices[0, 1]], 'k-', linewidth=2)
    
    # Add vertex labels
    plt.text(vertices[0, 0] - 0.1, vertices[0, 1] - 0.1, 'State 0', fontsize=14, ha='right', va='top')
    plt.text(vertices[1, 0] + 0.1, vertices[1, 1] - 0.1, 'State 1', fontsize=14, ha='left', va='top')
    plt.text(vertices[2, 0], vertices[2, 1] + 0.1, 'State 2', fontsize=14, ha='center', va='bottom')
    
    # Plot all points with colors based on the true belief state
    scatter = plt.scatter(
        belief_states_triangle[:, 0], 
        belief_states_triangle[:, 1], 
        c=belief_states_flat, 
        alpha=0.7,
        s=50,  # Increased point size
        edgecolors='black',  # Add black edges for better visibility
        linewidth=0.5
    )
    
    # Add title
    if not use_random:
        plt.title(f'True Belief States for All {total_sequences} Possible Sequences', pad=20, fontsize=16)
    else:
        plt.title(f'True Belief States for {total_sequences} Random Sequences', pad=20, fontsize=16)
    
    # Set the plot limits
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    
    # Remove axis labels and ticks
    plt.axis('off')
    
    # Save the 2D plot
    plt.savefig(os.path.join(save_dir, 'true_belief_states_control.png'), bbox_inches='tight', dpi=150)
    
    # Close the plot
    plt.close()
    
    # Create a 3D view of the belief states
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the 3D triangle
    triangle_3d = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    ax.plot_trisurf(triangle_3d[:, 0], triangle_3d[:, 1], triangle_3d[:, 2], 
                    color='lightgray', alpha=0.3)
    
    # Plot the points in 3D
    z_coords = 1 - belief_states_triangle[:, 0] - belief_states_triangle[:, 1]
    points_3d = np.column_stack([belief_states_triangle, z_coords])
    
    # Plot the points
    scatter = ax.scatter(
        points_3d[:, 0], 
        points_3d[:, 1], 
        points_3d[:, 2], 
        c=belief_states_flat, 
        alpha=0.7,
        s=50,  # Increased point size
        edgecolors='black',  # Add black edges for better visibility
        linewidth=0.5
    )
    
    # Set the view to be normal to the plane (1,1,1)
    ax.view_init(elev=35, azim=45)
    
    # Set the axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    
    # Add labels
    ax.set_xlabel('State 0', fontsize=12)
    ax.set_ylabel('State 1', fontsize=12)
    ax.set_zlabel('State 2', fontsize=12)
    
    # Add title
    if not use_random:
        plt.title(f'3D View of True Belief States for All {total_sequences} Possible Sequences', pad=20, fontsize=16)
    else:
        plt.title(f'3D View of True Belief States for {total_sequences} Random Sequences', pad=20, fontsize=16)
    
    # Save the 3D plot
    plt.savefig(os.path.join(save_dir, 'true_belief_states_control_3d.png'), bbox_inches='tight', dpi=150)
    
    # Close the plot
    plt.close()
    
    print(f"Control plots saved: {os.path.join(save_dir, 'true_belief_states_control.png')} and {os.path.join(save_dir, 'true_belief_states_control_3d.png')}")
    print(f"Generated {total_sequences} sequences of length {seq_length}")
    print(f"Total points plotted: {total_sequences * seq_length}")

def train_model(model, hmm_model, num_epochs=10, batch_size=32, seq_length=20, context_window=10, learning_rate=0.001, device='cuda', 
                save_interval=100, plot_interval=1000, num_epochs_per_save=10000):
    """Train the model with optimizations for GPU and large-scale training"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Create causal mask for transformer
    mask = torch.triu(torch.ones(context_window, context_window), diagonal=1).bool()
    mask = mask.to(device)
    
    # Training history
    train_losses = []
    val_losses = []
    
    # Create directory for checkpoints
    os.makedirs('checkpoints', exist_ok=True)
    
    # For large-scale training, we'll use a different approach
    total_epochs = num_epochs
    current_epoch = 0
    
    while current_epoch < total_epochs:
        model.train()
        
        # Generate batch of sequences from HMM
        sequences = hmm_model.generate_sequences(seq_length, batch_size)
        
        # Process each sequence in chunks of context_window size
        total_loss = 0
        num_chunks = 0
        
        # Variables to store residuals and true belief states for plotting
        residuals = None
        true_belief = None
        
        for i in range(0, seq_length - context_window + 1):
            # Extract a chunk of the sequence
            input_seq = sequences[:, i:i+context_window]
            target_seq = sequences[:, i+1:i+context_window+1]
            
            # Get true belief states
            true_belief_chunk = hmm_model.get_optimal_belief_states(input_seq)
            
            # Forward pass with causal mask
            output = model(input_seq, src_mask=mask)
            
            # Only get residuals at plot intervals
            if current_epoch % plot_interval == 0:
                # Get residuals
                residuals_chunk = model.get_residuals(input_seq, src_mask=mask)
                
                # Store for plotting
                if residuals is None:
                    residuals = residuals_chunk
                    true_belief = true_belief_chunk
                else:
                    # Concatenate along the sequence dimension
                    residuals = torch.cat([residuals, residuals_chunk], dim=1)
                    true_belief = torch.cat([true_belief, true_belief_chunk], dim=1)
            
            # Calculate loss
            output_flat = output.reshape(-1, output.size(-1))
            target_flat = target_seq.reshape(-1)
            
            # Ensure both tensors have the same batch size
            min_batch_size = min(output_flat.size(0), target_flat.size(0))
            output_flat = output_flat[:min_batch_size]
            target_flat = target_flat[:min_batch_size]
            
            loss = criterion(output_flat, target_flat)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_chunks += 1
        
        train_loss = total_loss / num_chunks
        
        # Only optimize projection matrix and plot at plot intervals
        if current_epoch % plot_interval == 0 and residuals is not None:
            # Optimize projection matrix and bias
            model.optimize_projection_matrix(residuals, true_belief)
            
            # Project residuals
            projected_residuals = model.project_residuals(residuals)
            
            # Plot belief states
            plot_belief_states(projected_residuals, true_belief, current_epoch)
        
        # Validation
        model.eval()
        with torch.no_grad():
            # Generate a new validation set
            val_sequences = hmm_model.generate_sequences(seq_length, batch_size)
            
            # For validation, we'll use the same approach as training
            val_total_loss = 0
            val_num_chunks = 0
            
            # Variables to store residuals and true belief states for plotting
            val_residuals = None
            val_true_belief = None
            
            for i in range(0, seq_length - context_window + 1):
                # Extract a chunk of the sequence
                val_input = val_sequences[:, i:i+context_window]
                val_target = val_sequences[:, i+1:i+context_window+1]
                
                # Get true belief states
                val_true_belief_chunk = hmm_model.get_optimal_belief_states(val_input)
                
                # Use the fresh validation set
                val_output = model(val_input, src_mask=mask)
                
                # Only get validation residuals at plot intervals
                if current_epoch % plot_interval == 0:
                    # Get validation residuals
                    val_residuals_chunk = model.get_residuals(val_input, src_mask=mask)
                    
                    # Store for plotting
                    if val_residuals is None:
                        val_residuals = val_residuals_chunk
                        val_true_belief = val_true_belief_chunk
                    else:
                        # Concatenate along the sequence dimension
                        val_residuals = torch.cat([val_residuals, val_residuals_chunk], dim=1)
                        val_true_belief = torch.cat([val_true_belief, val_true_belief_chunk], dim=1)
                
                # Calculate validation loss
                val_output_flat = val_output.reshape(-1, val_output.size(-1))
                val_target_flat = val_target.reshape(-1)
                
                # Ensure both tensors have the same batch size
                min_batch_size = min(val_output_flat.size(0), val_target_flat.size(0))
                val_output_flat = val_output_flat[:min_batch_size]
                val_target_flat = val_target_flat[:min_batch_size]
                
                val_loss_chunk = criterion(val_output_flat, val_target_flat)
                
                val_total_loss += val_loss_chunk.item()
                val_num_chunks += 1
            
            val_loss = val_total_loss / val_num_chunks
            
            # Plot belief states (only at specified intervals)
            if current_epoch % plot_interval == 0 and val_residuals is not None:
                # Project validation residuals
                val_projected_residuals = model.project_residuals(val_residuals)
                
                # Plot belief states
                plot_belief_states(val_projected_residuals, val_true_belief, current_epoch)
        
        # Record losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Print progress
        if current_epoch % 10 == 0:
            print(f'Epoch {current_epoch+1}/{total_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save checkpoint at specified intervals
        if (current_epoch + 1) % save_interval == 0:
            checkpoint_path = f'/content/checkpoints/model_epoch_{current_epoch+1}.pt'
            torch.save({
                'epoch': current_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')
        
        # For very large training runs, save intermediate results and reset history
        if (current_epoch + 1) % num_epochs_per_save == 0:
            # Save loss history
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.savefig(f'/content/plots/loss_history_epoch_{current_epoch+1}.png')
            plt.close()
            
            # Reset loss history to prevent memory issues
            train_losses = []
            val_losses = []
            
            # Save final model for this chunk
            final_checkpoint_path = f'/content/checkpoints/model_final_epoch_{current_epoch+1}.pt'
            torch.save({
                'epoch': current_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, final_checkpoint_path)
            print(f'Final checkpoint saved for epoch {current_epoch+1}: {final_checkpoint_path}')
        
        current_epoch += 1
    
    # Plot final training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('/content/plots/loss_history_final.png')
    plt.close()
    
    # Save final model
    final_model_path = '/content/checkpoints/model_final.pt'
    torch.save({
        'epoch': current_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, final_model_path)
    print(f'Final model saved: {final_model_path}')

def validate_belief_states(checkpoint_path, hmm_model, seq_length=10, batch_size=128, context_window=10, device='cuda', save_dir='plots'):
    """Run belief states validation on a given model checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        hmm_model: HMM model instance
        seq_length: Length of sequences to generate
        batch_size: Batch size for validation
        context_window: Context window size for transformer
        device: Device to run validation on
        save_dir: Directory to save plots
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Load the model checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create transformer model with the same parameters as in the main block
    model = TransformerWithResidual(
        vocab_size=3,  # Size of vocabulary (0, 1, 2)
        d_model=64,    # Dimension of model
        nhead=4,       # Number of attention heads
        num_layers=3,  # Number of transformer layers
        dim_feedforward=256,
        dropout=0.1
    )
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Get epoch from checkpoint
    epoch = checkpoint.get('epoch', 0)
    print(f"Validating model from epoch {epoch+1}")
    
    # Create causal mask for transformer
    mask = torch.triu(torch.ones(context_window, context_window), diagonal=1).bool()
    mask = mask.to(device)
    
    # Generate sequences from HMM
    print("Generating sequences for validation...")
    sequences = hmm_model.generate_sequences(seq_length, batch_size)
    print(f"Generated sequences shape: {sequences.shape}")
    
    # Process each sequence in chunks of context_window size
    residuals = None
    true_belief = None
    
    with torch.no_grad():
        for i in range(0, seq_length - context_window + 1):
            # Extract a chunk of the sequence
            input_seq = sequences[:, i:i+context_window]
            
            # Get true belief states
            true_belief_chunk = hmm_model.get_optimal_belief_states(input_seq)
            
            # Get residuals
            residuals_chunk = model.get_residuals(input_seq, src_mask=mask)
            
            # Debug prints
            print(f"\nChunk {i}:")
            print(f"Input sequence shape: {input_seq.shape}")
            print(f"True belief chunk shape: {true_belief_chunk.shape}")
            print(f"Residuals chunk shape: {residuals_chunk.shape}")
            print(f"True belief chunk contains NaN: {torch.isnan(true_belief_chunk).any()}")
            print(f"Residuals chunk contains NaN: {torch.isnan(residuals_chunk).any()}")
            
            # Store for plotting
            if residuals is None:
                residuals = residuals_chunk
                true_belief = true_belief_chunk
            else:
                # Concatenate along the sequence dimension
                residuals = torch.cat([residuals, residuals_chunk], dim=1)
                true_belief = torch.cat([true_belief, true_belief_chunk], dim=1)
    
    print("\nFinal shapes:")
    print(f"Residuals shape: {residuals.shape}")
    print(f"True belief shape: {true_belief.shape}")
    print(f"Residuals contains NaN: {torch.isnan(residuals).any()}")
    print(f"True belief contains NaN: {torch.isnan(true_belief).any()}")
    
    # Optimize projection matrix and bias
    print("\nOptimizing projection matrix...")
    model.optimize_projection_matrix(residuals, true_belief)
    
    # Project residuals
    projected_residuals = model.project_residuals(residuals)
    print(f"Projected residuals shape: {projected_residuals.shape}")
    print(f"Projected residuals contains NaN: {torch.isnan(projected_residuals).any()}")
    
    # Plot belief states
    print("\nPlotting belief states...")
    plot_belief_states(projected_residuals, true_belief, epoch, save_dir)
    
    print(f"Validation complete. Plots saved in {save_dir}")

    if __name__ == "__main__":
        # Determine device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Create HMM model
        hmm_model = HMMModel(device=device, model_type=HMM_MODEL_TYPE)
        print(f"Using HMM model type: {HMM_MODEL_TYPE}")

        # Check if a checkpoint path is provided as a command-line argument
        if len(sys.argv) > 1 and sys.argv[1] == "--validate":
            if len(sys.argv) < 3:
                print("Error: Please provide a checkpoint path after --validate")
                sys.exit(1)
            
            checkpoint_path = sys.argv[2]
            validate_belief_states(
                checkpoint_path=checkpoint_path,
                hmm_model=hmm_model,
                seq_length=SEQ_LENGTH,
                batch_size=BATCH_SIZE,
                context_window=CONTEXT_WINDOW,
                device=device
            )
            sys.exit(0)

        # Generate and plot all possible sequences as a control if requested
        if GENERATE_CONTROL_PLOT:
            print("Generating control plot of all possible sequences...")
            plot_all_possible_sequences(hmm_model, seq_length=SEQ_LENGTH, use_random=CONTROL_PLOT_RANDOM)

        # Create transformer model
        transformer_model = TransformerWithResidual(
            vocab_size=3,  # Size of vocabulary (0, 1, 2)
            d_model=64,    # Dimension of model
            nhead=1,       # Number of attention heads
            num_layers=4,  # Number of transformer layers
            dim_feedforward=256,
            dropout=0.1
        )

        # Set default save and plot intervals if not provided
        save_interval = SAVE_INTERVAL
        if save_interval is None:
            save_interval = NUM_EPOCHS//100 if NUM_EPOCHS > 1e4 else 10

        plot_interval = PLOT_INTERVAL
        if plot_interval is None:
            plot_interval = save_interval

        # Train the model
        train_model(
            transformer_model, 
            hmm_model, 
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            seq_length=SEQ_LENGTH,
            context_window=CONTEXT_WINDOW,
            learning_rate=LEARNING_RATE, 
            device=device,
            save_interval=save_interval,
            plot_interval=plot_interval
        ) 