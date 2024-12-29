import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Custom Dataset class for MNIST
class MNISTDataset(Dataset):
    def __init__(self, csv_file):
        # Read CSV file
        data = pd.read_csv(csv_file)

        # Separate labels and features
        self.labels = data.iloc[:, 0].values
        self.features = data.iloc[:, 1:].values

        # First normalize to [0,1]
        self.features = self.features / 255.0

        # Then center the data
        self.features = self.features - np.mean(self.features, axis=0)

        # Rescale to [0,1] after centering
        min_vals = np.min(self.features, axis=0)
        max_vals = np.max(self.features, axis=0)
        self.features = (self.features - min_vals) / (max_vals - min_vals + 1e-8)

        # Convert to tensors
        self.features = torch.FloatTensor(self.features)
        self.labels = torch.LongTensor(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self, input_size=784, latent_size=128):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Linear(input_size, latent_size, bias=False)

        # Decoder
        self.decoder = nn.Linear(latent_size, input_size, bias=False)

        # Initialize weights
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)

    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)

        # Encode
        encoded = self.encoder(x)

        # Decode
        decoded = torch.sigmoid(self.decoder(encoded))

        return decoded


def train_model(model, train_loader, num_epochs=40, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Lists to store losses
    train_losses = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for data, _ in train_loader:
            data = data.to(device)

            # Forward pass
            output = model(data)
            loss = criterion(output, data.view(data.size(0), -1))

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # Print epoch statistics
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.6f}')

        # Compare encoder weights with PCA matrix
        if epoch % 10 == 0:
            compare_weights_with_pca(model.encoder.weight.data)

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()


def compare_weights_with_pca(encoder_weights):
    # Here you would compare with VL matrix from PCA
    print(f"Encoder weights shape: {encoder_weights.shape}")


def visualize_reconstructions(model, test_loader, num_images=5):
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        # Get a batch of test data
        data, _ = next(iter(test_loader))
        data = data[:num_images].to(device)

        # Get reconstructions
        reconstructions = model(data)

        # Plot original and reconstructed images
        plt.figure(figsize=(12, 4))
        for i in range(num_images):
            # Original
            plt.subplot(2, num_images, i + 1)
            plt.imshow(data[i].cpu().view(28, 28), cmap='gray')
            plt.title('Original')
            plt.axis('off')

            # Reconstruction
            plt.subplot(2, num_images, i + num_images + 1)
            plt.imshow(reconstructions[i].cpu().view(28, 28), cmap='gray')
            plt.title('Reconstructed')
            plt.axis('off')

        plt.tight_layout()
        plt.show()


def main():
    # Hyperparameters
    BATCH_SIZE = 250
    LATENT_SIZE = 128
    NUM_EPOCHS = 40

    # Load datasets
    print("Loading datasets...")
    train_dataset = MNISTDataset('mnist_train.csv')
    test_dataset = MNISTDataset('mnist_test.csv')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("Datasets loaded successfully")

    # Initialize model
    print("Initializing model...")
    model = Autoencoder(input_size=784, latent_size=LATENT_SIZE)

    # Train model
    print("Starting training...")
    train_model(model, train_loader, NUM_EPOCHS)

    # Visualize results
    print("Generating visualizations...")
    visualize_reconstructions(model, test_loader)

    print("Training completed!")


if __name__ == "__main__":
    main()