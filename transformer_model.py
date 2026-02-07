import torch
import torch.nn as nn

class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_dim=5,
        model_dim=64,
        num_heads=4,
        num_layers=2,
        horizon=12,
        output_dim=3
    ):
        super(TimeSeriesTransformer, self).__init__()

        # input embedding
        self.embedding = nn.Linear(input_dim, model_dim)

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # output layer
        self.fc = nn.Linear(model_dim, horizon * output_dim)

        self.horizon = horizon
        self.output_dim = output_dim

    def forward(self, x):
        x shape: (batch_size, seq_len, input_dim)
        x = self.embedding(x)
        x = self.transformer(x)

        # take last time step
        x = x[:, -1, :]

        x = self.fc(x)
        return x.view(-1, self.horizon, self.output_dim)
    
