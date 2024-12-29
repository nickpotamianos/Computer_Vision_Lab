import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from autoencoder_mnist_4 import MNISTDataset


class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[512, 256], latent_dim=2):
        super(VAE, self).__init__()

        # Encoder layers
        encoder_layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, hidden_dim, bias=False),
                nn.ReLU()
            ])
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Mean and variance layers
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim, bias=False)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim, bias=False)

        # Decoder layers
        decoder_layers = []
        hidden_dims.reverse()
        in_dim = latent_dim
        for hidden_dim in hidden_dims:
            decoder_layers.extend([
                nn.Linear(in_dim, hidden_dim, bias=False),
                nn.ReLU()
            ])
            in_dim = hidden_dim
        decoder_layers.extend([
            nn.Linear(hidden_dims[-1], input_dim, bias=False),
            nn.Sigmoid()
        ])
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    # Binary Cross Entropy loss
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL Divergence loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train_epoch(model, train_loader, optimizer, device, epoch, fixed_noise):
    model.train()
    train_loss = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Generate samples from fixed noise at specified epochs
    if epoch in [1, 50, 100]:
        visualize_fixed_noise_samples(model, fixed_noise, device, epoch)

    return train_loss / len(train_loader.dataset)


def visualize_fixed_noise_samples(model, fixed_noise, device, epoch):
    model.eval()
    with torch.no_grad():
        samples = model.decode(fixed_noise).cpu()

        plt.figure(figsize=(8, 8))
        for i in range(64):
            plt.subplot(8, 8, i + 1)
            plt.imshow(samples[i].reshape(28, 28), cmap='gray')
            plt.axis('off')
        plt.suptitle(f'Fixed Noise Samples - Epoch {epoch}')
        plt.tight_layout()
        plt.show()


def visualize_latent_space(model, test_loader, device):
    model.eval()
    z_points = []
    labels = []

    with torch.no_grad():
        for data, y in test_loader:
            data = data.to(device)
            mu, _ = model.encode(data)
            z_points.append(mu.cpu().numpy())
            labels.append(y.numpy())

    z_points = np.concatenate(z_points, axis=0)
    labels = np.concatenate(labels, axis=0)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z_points[:, 0], z_points[:, 1], c=labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title('Latent Space Visualization')
    plt.xlabel('z[0]')
    plt.ylabel('z[1]')
    plt.show()


def main():
    # Hyperparameters as specified in the exercise
    batch_size = 250
    epochs = 100
    latent_dim = 2
    learning_rate = 1e-3

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load datasets
    train_dataset = MNISTDataset('mnist_train.csv')
    test_dataset = MNISTDataset('mnist_test.csv')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model and optimizer
    model = VAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Generate fixed noise for visualization
    fixed_noise = torch.randn(64, latent_dim).to(device)

    # Training loop
    train_losses = []
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device, epoch, fixed_noise)
        train_losses.append(loss)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Average loss = {loss:.4f}')

    # Final visualizations
    visualize_latent_space(model, test_loader, device)

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()