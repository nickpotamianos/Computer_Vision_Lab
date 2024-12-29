import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# 1) Define the Kernel Mapping
# -----------------------------
def kernel_transform(X, alpha=0.1):
    """
    Apply the nonlinear transformation k(x) = exp(-x^2 / alpha)
    component-wise to each entry of X.
    X is of shape [N, M] (N samples, M features/pixels).
    Return a tensor of the same shape.
    """
    return torch.exp(-(X ** 2) / alpha)


def kernel_inverse_transform(X_k, alpha=0.1):
    """
    Attempt a naive per-pixel inversion of
      k(x_j) = exp(-x_j^2 / alpha)
    => x_j^2 = - alpha * ln(k(x_j))
    => x_j   = sqrt(- alpha * ln(k(x_j)))   [ignoring sign]

    Note: This is only well-defined if k(x_j) is in (0,1].
    """
    # Clamp values to avoid log(0)
    epsilon = 1e-10
    X_k_clamped = torch.clamp(X_k, min=epsilon, max=1.0)
    return torch.sqrt(-alpha * torch.log(X_k_clamped))


# --------------------------------
# 2) A simple PCA class (same as in Part 1)
# --------------------------------
class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Center the data
        self.mean = torch.mean(X, dim=0)
        X_centered = X - self.mean

        # Covariance = X^T X / (N - 1)
        N = X.shape[0]
        cov_matrix = torch.matmul(X_centered.T, X_centered) / (N - 1)

        # Eigen-decomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

        # Sort descending
        idx = torch.argsort(eigenvalues, descending=True)
        self.components = eigenvectors[:, idx[:self.n_components]]

        return self

    def transform(self, X):
        X_centered = X - self.mean
        return torch.matmul(X_centered, self.components)

    def inverse_transform(self, Z):
        """
        Inverse of transform in the *feature* space of dimension M
        back to the original feature space dimension M.
        """
        return torch.matmul(Z, self.components.T) + self.mean


# -----------------------------------------------------
# 3) Functions to compute mean/cov in the kernel space
# -----------------------------------------------------
def calculate_mean_digit_kspace(X_k, y, digit):
    """
    Compute mean of digit `digit` in the kernel-transformed space.
    """
    mask = (y == digit)
    digit_data_k = X_k[mask]
    return digit_data_k.mean(dim=0)


def calculate_covariance_matrix_kspace(X_k):
    """
    Just the standard sample covariance for the kernel-mapped data.
    """
    Xk_centered = X_k - X_k.mean(dim=0)
    N = X_k.shape[0]
    return torch.matmul(Xk_centered.T, Xk_centered) / (N - 1)


# ------------------------------------------------
# 4) Plotting helpers
# ------------------------------------------------
def plot_digit(data, title):
    if torch.is_tensor(data):
        data = data.numpy()
    plt.imshow(data.reshape(28, 28), cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


def plot_principal_components(pca_components, n_components=8):
    """
    pca_components: shape [M, n_components]
    We'll reshape each to 28x28 to visualize.
    But remember these are in the *kernel space* now!
    """
    comps = pca_components.T  # shape => [n_components, M]
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.ravel()

    for i in range(n_components):
        img = comps[i].reshape(28, 28)
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'PC {i + 1}')

    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------------
# 5) Reconstruct and plot digits for different numbers of components in *kernel* space
# --------------------------------------------------------------------------------
def reconstruct_and_plot_kernel(X, dimensions, digit_idx=0):
    """
    X is the kernel-transformed data (shape [N, 784])
    We'll do PCA in kernel space, reconstruct in kernel space,
    then apply inverse transform to get back a naive "digit" in original space.
    """
    original_k = X[digit_idx]  # single row in kernel space

    fig, axes = plt.subplots(1, len(dimensions), figsize=(20, 4))
    for i, dim in enumerate(dimensions):
        pca = PCA(n_components=dim)
        pca.fit(X)
        Z = pca.transform(X)  # shape [N, dim]
        Xk_reconstructed = pca.inverse_transform(Z)  # shape [N, 784]

        # Inverse the kernel transform back to original space
        digit_reconstructed = kernel_inverse_transform(Xk_reconstructed[digit_idx].unsqueeze(0))
        digit_reconstructed = digit_reconstructed.squeeze(0).cpu().numpy()

        axes[i].imshow(digit_reconstructed.reshape(28, 28), cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'L = {dim}')

    plt.suptitle('Digit Reconstructions (Kernel Space -> Inverse Kernel)')
    plt.show()


def plot_error_histogram_kernel(X, dimensions):
    """
    Plot reconstruction error histograms in the *original space*
    for different L by:
      1) PCA in kernel space
      2) Inverse from kernel space to original space
      3) Compare to the original images (also inversely mapped for 'original' reference).
    """
    # First, invert the *original* kernel data to get baseline images in original space
    X_inverted_orig = kernel_inverse_transform(X)

    plt.figure(figsize=(12, 6))

    for dim in dimensions:
        pca = PCA(n_components=dim)
        pca.fit(X)
        Z = pca.transform(X)
        Xk_reconstructed = pca.inverse_transform(Z)
        X_reconstructed_orig = kernel_inverse_transform(Xk_reconstructed)

        # Compute MSE in original space
        errors = torch.mean((X_inverted_orig - X_reconstructed_orig) ** 2, dim=1)
        plt.hist(errors.cpu().numpy(), bins=50, alpha=0.5, label=f'L={dim}')

    plt.xlabel('Mean Squared Error (original space)')
    plt.ylabel('Frequency')
    plt.title('Reconstruction Error Histogram (Kernel Space PCA)')
    plt.legend()
    plt.show()


# -------------------------------------------
# 6) Putting it all together
# -------------------------------------------
def main_kernel_pca_example():
    # Load your data (the same as Part 1)
    # For demonstration, let's pick digits 0 and 1
    train_path = 'mnist_train.csv'
    test_path = 'mnist_test.csv'
    selected_digits = [0, 1]

    # Load data
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Separate labels and features
    X_train = train_data.drop('label', axis=1).values
    y_train = train_data['label'].values

    # Filter
    mask = np.isin(y_train, selected_digits)
    X_train = X_train[mask]
    y_train = y_train[mask]

    # Tensor + normalize
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32) / 255.
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    # ---------------------------
    # Move to kernel space
    # ---------------------------
    alpha = 0.1
    X_train_k = kernel_transform(X_train_tensor, alpha=alpha)

    for digit in selected_digits:
        print(f"\n--- Analyzing digit {digit} in KERNEL space ---")

        # (a') mean digit in kernel space
        mean_digit_k = calculate_mean_digit_kspace(X_train_k, y_train_tensor, digit)
        # Convert back to original space (naive) to visualize
        mean_digit_orig_approx = kernel_inverse_transform(mean_digit_k.unsqueeze(0), alpha=alpha).squeeze(0)
        plot_digit(mean_digit_orig_approx, title=f"Mean Digit {digit} (Kernel->Inverse)")

        # (b') covariance matrix in kernel space
        digit_data_k = X_train_k[y_train_tensor == digit]
        cov_k = calculate_covariance_matrix_kspace(digit_data_k)
        print(f"Covariance matrix shape (kernel space): {cov_k.shape}")

        # (c') first 8 principal components in kernel space
        pca_k = PCA(n_components=8)
        pca_k.fit(digit_data_k)
        print("Components shape in kernel space:", pca_k.components.shape)
        plot_principal_components(pca_k.components, n_components=8)

        # (d') Digit reconstructions for L=1,8,16,64,256
        # We'll pick the first sample of the chosen digit for reconstruction
        dims = [1, 8, 16, 64, 256,1024]
        reconstruct_and_plot_kernel(digit_data_k, dimensions=dims, digit_idx=0)

        # (e') Error histograms
        plot_error_histogram_kernel(digit_data_k, dimensions=dims)


if __name__ == "__main__":
    main_kernel_pca_example()
