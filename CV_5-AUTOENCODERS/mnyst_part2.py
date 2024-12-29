import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None  # This will be our VL matrix
        self.mean = None

    def fit(self, X):
        # Convert to PyTorch tensor if not already
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)

        # Center the data
        self.mean = torch.mean(X, dim=0)
        X_centered = X - self.mean

        # Compute covariance matrix
        N = X.shape[0]
        cov_matrix = torch.matmul(X_centered.T, X_centered) / (N - 1)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

        # Sort eigenvalues and corresponding eigenvectors in descending order
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvectors = eigenvectors[:, idx]

        # Store first n_components eigenvectors as VL matrix
        self.components = eigenvectors[:, :self.n_components]
        return self

    def transform(self, X):
        """Transform data into compressed representation"""
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        X_centered = X - self.mean
        return torch.matmul(X_centered, self.components)

    def inverse_transform(self, X_transformed):
        """Reconstruct original data from compressed representation"""
        return torch.matmul(X_transformed, self.components.T) + self.mean


def load_and_preprocess_data(train_path, test_path):
    """Load and preprocess MNIST data"""
    # Load data
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Separate labels and features
    X_train = train_data.drop('label', axis=1).values
    y_train = train_data['label'].values
    X_test = test_data.drop('label', axis=1).values
    y_test = test_data['label'].values

    # Convert to PyTorch tensors and normalize to [0,1]
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32) / 255.0
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32) / 255.0
    y_train_tensor = torch.tensor(y_train)
    y_test_tensor = torch.tensor(y_test)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor


def plot_characteristic_pairs(original, reconstructed, labels, title):
    """Plot characteristic pairs of original and reconstructed images"""
    plt.figure(figsize=(15, 5))

    for i in range(3):  # Show 3 characteristic examples
        plt.subplot(2, 3, i + 1)
        plt.imshow(original[i].reshape(28, 28), cmap='gray')
        plt.title(f'Original (Digit {labels[i]})')
        plt.axis('off')

        plt.subplot(2, 3, i + 4)
        plt.imshow(reconstructed[i].reshape(28, 28), cmap='gray')
        plt.title(f'Compressed (Digit {labels[i]})')
        plt.axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    print("Loading MNIST data...")
    X_train, y_train, X_test, y_test = load_and_preprocess_data('mnist_train.csv', 'mnist_test.csv')

    # Part 2: Apply PCA with L=128 and compute VL matrix
    print("Applying PCA with L=128...")
    pca = PCA(n_components=128)
    pca.fit(X_train)

    # Print information about VL matrix
    print(f"\nVL matrix shape: {pca.components.shape}")
    print("First few values of VL matrix:")
    print(pca.components[:5, :5])

    # Part (α): Reconstruct compressed training data
    print("\nCompressing and reconstructing training data...")
    X_train_transformed = pca.transform(X_train)
    X_train_reconstructed = pca.inverse_transform(X_train_transformed)

    # Part (β): Generate compressed test data
    print("Compressing and reconstructing test data...")
    X_test_transformed = pca.transform(X_test)
    X_test_reconstructed = pca.inverse_transform(X_test_transformed)

    # Calculate compression ratio and reconstruction error
    original_dims = X_train.shape[1]
    compressed_dims = pca.n_components
    compression_ratio = original_dims / compressed_dims

    train_mse = torch.mean((X_train - X_train_reconstructed) ** 2)
    test_mse = torch.mean((X_test - X_test_reconstructed) ** 2)

    print(f"\nCompression Statistics:")
    print(f"Original dimensions: {original_dims}")
    print(f"Compressed dimensions: {compressed_dims}")
    print(f"Compression ratio: {compression_ratio:.2f}:1")
    print(f"Training data MSE: {train_mse:.6f}")
    print(f"Test data MSE: {test_mse:.6f}")

    # Show characteristic pairs for both training and test data
    # Select some diverse examples
    digits_to_show = [0, 4, 9]  # Different digit shapes
    for digit in digits_to_show:
        # Training data
        train_indices = torch.where(y_train == digit)[0][:3]
        plot_characteristic_pairs(
            X_train[train_indices],
            X_train_reconstructed[train_indices],
            [digit] * 3,
            f'Training Data Compression Results - Digit {digit}'
        )

        # Test data
        test_indices = torch.where(y_test == digit)[0][:3]
        plot_characteristic_pairs(
            X_test[test_indices],
            X_test_reconstructed[test_indices],
            [digit] * 3,
            f'Test Data Compression Results - Digit {digit}'
        )


if __name__ == "__main__":
    main()