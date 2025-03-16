import numpy as np
import torch
import torch.nn as nn


class SparrowMahjongLSTM(nn.Module):
    def __init__(
            self,
            input_size=37,
            hidden_size=128,
            output_size=6,
            num_lstm_layers=1,
            num_fc_layers=2,
            fc_hidden_size=64,
    ):
        super(SparrowMahjongLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_lstm_layers,
            batch_first=True,
        )

        # Fully connected layers
        fc_layers = []
        input_dim = hidden_size

        for _ in range(num_fc_layers):
            fc_layers.append(nn.Linear(input_dim, fc_hidden_size))
            fc_layers.append(nn.ReLU())
            input_dim = fc_hidden_size

        self.fc_layers = nn.Sequential(*fc_layers)

        # Output layer
        self.output_layer = nn.Linear(input_dim, output_size)

    def forward(self, x, hidden):
        """
        x: Tensor of shape (batch_size, seq_length, input_size)
        hidden: Tuple containing the initial hidden state and cell state,
                each of shape (num_layers, batch_size, hidden_size)
        """
        out, hidden = self.lstm(x, hidden)
        out = out[:, -1, :]  # Get the output from the last time step
        out = self.fc_layers(out)
        out = self.output_layer(out)
        return out, hidden

    def init_hidden(self, batch_size):
        # Initialize hidden state and cell state with zeros
        device = next(self.parameters()).device  # Ensure the hidden states are on the same device as the model
        hidden = (
            torch.zeros(self.num_lstm_layers, batch_size, self.hidden_size, device=device),
            torch.zeros(self.num_lstm_layers, batch_size, self.hidden_size, device=device),
        )
        return hidden

    def get_model_weights(self):
        """Returns a flat numpy array of all model weights (including biases)."""
        weights = []
        for param in self.parameters():
            weights.append(param.detach().cpu().numpy().flatten())
        return np.concatenate(weights)

    def set_model_weights(self, flat_weights):
        """Sets the model's weights from a flat numpy array."""
        prev_ind = 0
        for param in self.parameters():
            flat_size = param.numel()
            new_weights = flat_weights[prev_ind:prev_ind + flat_size].reshape(param.shape)
            param.data.copy_(torch.from_numpy(new_weights))
            prev_ind += flat_size

    def set_random_weights_from_model(self):
        """Set random weights to the model, with the same shape as the current model weights."""
        model_weights = self.get_model_weights()  # Get the current model weights
        random_weights = np.random.randn(len(model_weights))  # Generate random weights
        self.set_model_weights(random_weights)  # Set the random weights to the model
