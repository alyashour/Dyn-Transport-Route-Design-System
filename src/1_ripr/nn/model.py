import torch
import torch.nn as nn


class ShotgunPredictor(nn.Module):
    def __init__(self, input_dim=60, num_stops=1952):
        super().__init__()
        
        # Output dimension is N * N (all possible OD pairs)
        self.output_dim = num_stops * num_stops
        
        # Single Layer Architecture
        # Direct projection from Input Features (approx 60) -> Output Grid (N^2)
        # This drastically reduces RAM usage compared to hidden layers
        self.net = nn.Sequential(
            nn.Linear(input_dim, self.output_dim),
            # We use LogSoftmax because KLDivLoss expects log-probabilities
            nn.LogSoftmax(dim=1) 
        )

    def forward(self, x):
        return self.net(x)


class SMGPredictor(nn.Module):
    def __init__(self, input_dim=60, num_stops=1952, rank=32, hidden_dim=128):
        super().__init__()
        self.num_stops = num_stops
        self.rank = rank
        
        # 1. THE ENCODER (Hidden Layer)
        # This breaks the "linear floor". It allows the model to mix features 
        # (e.g. Month + DayOfWeek) before deciding on traffic patterns.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.25) # Prevents overfitting to specific dates
        )
        
        # 2. THE FACTOR HEADS
        # Instead of predicting N*N (Million) numbers directly, we predict 
        # N*Rank (Thousands) numbers for Origins and Destinations separately.
        
        # Head A: "How much traffic starts here?"
        # Output shape: num_stops * rank
        self.head_origin = nn.Linear(hidden_dim, num_stops * rank)
        
        # Head B: "How much traffic ends here?"
        # Output shape: num_stops * rank
        self.head_dest = nn.Linear(hidden_dim, num_stops * rank)
        
        # 3. STATIC BIAS
        # A learnable matrix representing the "Average Day".
        # The model only has to learn how today *differs* from the average.
        self.static_bias = nn.Parameter(torch.zeros(num_stops, num_stops))

    def forward(self, x):
        batch_size = x.shape[0]
        
        # 1. Encode Features
        # x: (Batch, 60) -> features: (Batch, 128)
        features = self.encoder(x)
        
        # 2. Predict Factors
        # We reshape linear output into (Batch, Stops, Rank)
        origins = self.head_origin(features).view(batch_size, self.num_stops, self.rank)
        dests = self.head_dest(features).view(batch_size, self.num_stops, self.rank)
        
        # 3. Reconstruct Grid (The "Netflix" Trick)
        # We multiply Origin factors by Destination factors to get the full grid.
        # (B, N, Rank) @ (B, Rank, N) -> (B, N, N)
        grid_logits = torch.bmm(origins, dests.permute(0, 2, 1))
        
        # Add the static average grid
        grid_logits = grid_logits + self.static_bias
        
        # 4. Flatten and Softmax
        # Flatten from (Batch, N, N) to (Batch, N*N) for the loss function
        flat_logits = grid_logits.view(batch_size, -1)
        
        return torch.log_softmax(flat_logits, dim=1)

