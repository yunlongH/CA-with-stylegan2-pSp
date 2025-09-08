import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, opts):
        """
        Discriminator that processes (batch, 18, 512) → (batch, 1)
        Args:
            opts: Configuration object with attributes:
                  - opts.latent_dim: Latent space dimension (e.g., 512)
                  - opts.n_layer_disc: Number of layers in the MLP
        """
        super(Discriminator, self).__init__()
        self.opts = opts
        input_dim = self.opts.latent_dim  # 512
        hidden_dim = 512
        num_layers = self.opts.n_layer_disc

        layers = []
        prev_dim = input_dim

        # Dynamically add hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())  # Activation function
            prev_dim = hidden_dim  # Update previous dimension

        # Final layer (hidden_dim -> 1)
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())  # Output probability

        # Define MLP
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: Tensor of shape (batch, 18, 512)
        """
        x = self.mlp(x)  # Shape: (batch, 18, 1)
        x = x.mean(dim=1)  # Aggregate across 18 features → (batch, 1)
        return x



class DiscriminatorGlobal(nn.Module):
    def __init__(self, opts):
        """
        DiscriminatorGlobal processes flattened (batch, 18*512) → (batch, 1)
        If num_layers = 1, it acts as a logistic regression model (no hidden layers).
        
        Args:
            opts: Configuration object with attributes:
                  - opts.style_dim: Style space dimension (e.g., 18)
                  - opts.latent_dim: Latent space dimension (e.g., 512)
                  - opts.n_layer_disc: Number of layers in the MLP
        """
        super(DiscriminatorGlobal, self).__init__()
        self.opts = opts

        input_dim = self.opts.style_dim * self.opts.latent_dim  # 18 * 512 = 9216
        hidden_dim = 512
        num_layers = self.opts.n_layer_disc

        layers = []
        
        if num_layers == 1:
            # Logistic regression: Direct mapping from input_dim → 1
            layers.append(nn.Linear(input_dim, 1))
            layers.append(nn.Sigmoid())  # Output probability
        else:
            # MLP with multiple layers
            prev_dim = input_dim
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())  # Activation function
                prev_dim = hidden_dim  # Update previous dimension

            # Final layer (hidden_dim -> 1)
            layers.append(nn.Linear(hidden_dim, 1))
            layers.append(nn.Sigmoid())  # Output probability

        # Define MLP
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: Tensor of shape (batch, 18, 512)
        """
        x = x.view(x.shape[0], -1)  # Flatten to (batch, 9216)
        x = self.mlp(x)  # Shape: (batch, 1)
        return x

class CustomLatentClassifier(nn.Module):
    def __init__(self, input_dim=512, num_layers=1, hidden_dim=None):
        """
        Classifier for latent vectors (batch_size x input_dim).

        Args:
            input_dim (int): Dimension of input features (default: 512).
            num_layers (int): Number of hidden layers (1 = logistic regression, up to 5 layers).
            hidden_dim (int or None): Automatically determined if None.
        """
        super(CustomLatentClassifier, self).__init__()

        if num_layers < 1 or num_layers > 5:
            raise ValueError("num_layers must be between 1 and 5.")

        if hidden_dim is None:
            hidden_dim1 = input_dim // 2
            hidden_dim2 = hidden_dim1 // 2
            hidden_dim3 = hidden_dim2 // 2
            hidden_dim4 = hidden_dim3 // 2
        else:
            hidden_dim1 = hidden_dim
            hidden_dim2 = hidden_dim1 // 2
            hidden_dim3 = hidden_dim2 // 2
            hidden_dim4 = hidden_dim3 // 2

        layers = []

        if num_layers == 1:
            layers.append(nn.Linear(input_dim, 1))
        elif num_layers == 2:
            layers += [nn.Linear(input_dim, hidden_dim1), nn.ReLU(),
                       nn.Linear(hidden_dim1, 1)]
        elif num_layers == 3:
            layers += [nn.Linear(input_dim, hidden_dim1), nn.ReLU(),
                       nn.Linear(hidden_dim1, hidden_dim2), nn.ReLU(),
                       nn.Linear(hidden_dim2, 1)]
        elif num_layers == 4:
            layers += [nn.Linear(input_dim, hidden_dim1), nn.ReLU(),
                       nn.Linear(hidden_dim1, hidden_dim2), nn.ReLU(),
                       nn.Linear(hidden_dim2, hidden_dim3), nn.ReLU(),
                       nn.Linear(hidden_dim3, 1)]
        elif num_layers == 5:
            layers += [nn.Linear(input_dim, hidden_dim1), nn.ReLU(),
                       nn.Linear(hidden_dim1, hidden_dim2), nn.ReLU(),
                       nn.Linear(hidden_dim2, hidden_dim3), nn.ReLU(),
                       nn.Linear(hidden_dim3, hidden_dim4), nn.ReLU(),
                       nn.Linear(hidden_dim4, 1)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze(-1)  # (batch_size,) for BCEWithLogitsLoss




