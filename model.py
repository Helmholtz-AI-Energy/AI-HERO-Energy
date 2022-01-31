import torch
from torch import nn


class LoadForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 num_layer: int = 1, dropout: float = 0, batch_first: bool = True, device: torch.device = None):
        super(LoadForecaster, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layer
        self.device = device

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layer, dropout=dropout, batch_first=batch_first,
                            device=self.device)
        self.fully_connected = nn.Linear(hidden_size, output_size, device=self.device)

    def forward(self, input_sequence, hidden):
        output, hidden = self.lstm(input_sequence, hidden)
        output = self.fully_connected(output)
        return output, hidden

    def init_hidden(self, batch_size):
        hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        cell_state = torch.randn(self.num_layers, batch_size, self.hidden_size, device=self.device)

        return hidden_state, cell_state
