import torch
import torch.nn as nn
import torch.nn.functional as F

class ActionNetwork(nn.Module):
    def __init__(self, lstm_hidden_size=256, embedding_dim=64):
        super(ActionNetwork, self).__init__()

        # Dimensions
        self.action_dim = 2  # 3 possible actions: Move, Stay, Zap
        self.unit_dim = 16   # 16x1 for target units
        self.coordinate_dim = 2  # 1D output for each coordinate

        # LSTM for processing state
        self.lstm = nn.LSTM(input_size=embedding_dim,
                          hidden_size=lstm_hidden_size,
                          batch_first=True)

        # Action ID path
        self.action_embedding = nn.Embedding(self.action_dim, embedding_dim)
        self.action_fc = nn.Linear(lstm_hidden_size, embedding_dim)

        # Target unit path
        self.unit_embedding = nn.Embedding(self.unit_dim, embedding_dim)
        self.unit_fc = nn.Linear(lstm_hidden_size, embedding_dim)
        self.unit_attention = nn.Linear(embedding_dim, 1)

        # Coordinate paths (X and Y offsets)
        self.offset_x_fc = nn.Linear(lstm_hidden_size, self.coordinate_dim)
        self.offset_y_fc = nn.Linear(lstm_hidden_size, self.coordinate_dim)

    def forward(self, state, available_actions, available_units):
        """
        Args:
            state: Input state tensor (batch_size, seq_len, input_dim)
            available_actions: Boolean mask of available actions (batch_size, action_dim)
            available_units: Boolean mask of available units (batch_size, unit_dim)
        """
        # Process state through LSTM
        lstm_out, _ = self.lstm(state)
        lstm_final = lstm_out[:, -1, :]  # Take final LSTM state

        # Action ID selection
        action_embeds = self.action_embedding(torch.arange(self.action_dim).to(state.device))
        action_query = self.action_fc(lstm_final)

        action_scores = torch.matmul(action_query, action_embeds.t())
        action_scores = action_scores.masked_fill(~available_actions, float('-inf'))
        action_probs = F.softmax(action_scores, dim=-1)

        # Target unit selection
        unit_embeds = self.unit_embedding(torch.arange(self.unit_dim).to(state.device))
        unit_query = self.unit_fc(lstm_final)

        # Attention mechanism
        unit_keys = torch.tanh(unit_embeds @ unit_query.unsqueeze(-1))
        unit_scores = self.unit_attention(unit_keys).squeeze(-1)
        unit_scores = unit_scores.masked_fill(~available_units, float('-inf'))
        unit_probs = F.softmax(unit_scores, dim=-1)

        # Coordinate offsets
        offset_x = self.offset_x_fc(lstm_final)
        offset_y = self.offset_y_fc(lstm_final)

        return {
            'action_probs': action_probs,
            'unit_probs': unit_probs,
            'offset_x': offset_x,
            'offset_y': offset_y
        }

    def sample_action(self, probs):
        """Sample from probability distribution using argmax or sampling"""
        if self.training:
            return torch.multinomial(probs, 1)
        else:
            return torch.argmax(probs, dim=-1, keepdim=True)

class ActionLoss(nn.Module):
    def __init__(self, action_weight=1.0, unit_weight=1.0, coord_weight=1.0):
        super(ActionLoss, self).__init__()
        self.action_weight = action_weight
        self.unit_weight = unit_weight
        self.coord_weight = coord_weight

    def forward(self, outputs, actions_taken, units_selected, coords_taken, rewards):
        """
        Calculate loss using rewards/points as supervision signal

        Args:
            outputs: Dictionary containing network outputs
            actions_taken: Indices of actions that were taken (batch_size, 1)
            units_selected: Indices of units that were selected (batch_size, 1)
            coords_taken: Coordinates that were chosen (batch_size, 2)
            rewards: Reward signals for each action (batch_size, 1)
        """
        batch_size = rewards.size(0)

        # Action loss (policy gradient with rewards)
        action_log_probs = torch.log(outputs['action_probs'].gather(1, actions_taken))
        action_loss = -self.action_weight * (action_log_probs * rewards).mean()

        # Unit selection loss
        unit_log_probs = torch.log(outputs['unit_probs'].gather(1, units_selected))
        unit_loss = -self.unit_weight * (unit_log_probs * rewards).mean()

        # Coordinate loss (MSE weighted by rewards)
        coord_pred = torch.cat([outputs['offset_x'], outputs['offset_y']], dim=1)
        coord_loss = self.coord_weight * (F.mse_loss(coord_pred, coords_taken, reduction='none') * rewards).mean()

        total_loss = action_loss + unit_loss + coord_loss

        return {
            'total_loss': total_loss,
            'action_loss': action_loss,
            'unit_loss': unit_loss,
            'coord_loss': coord_loss
        }

# Example usage
def main():
    batch_size = 32
    seq_len = 10
    input_dim = 64

    model = ActionNetwork()
    criterion = ActionLoss(action_weight=1.0, unit_weight=0.8, coord_weight=0.5)

    # Sample inputs
    state = torch.randn(batch_size, seq_len, input_dim)
    available_actions = torch.ones(batch_size, 3, dtype=torch.bool)
    available_units = torch.ones(batch_size, 16, dtype=torch.bool)

    # Forward pass
    outputs = model(state, available_actions, available_units)

    # Sample some actions
    actions_taken = model.sample_action(outputs['action_probs'])
    units_selected = model.sample_action(outputs['unit_probs'])
    coords_taken = torch.randn(batch_size, 2)  # Simulated coordinates

    # Simulated rewards
    rewards = torch.randn(batch_size, 1)  # Could be points/rewards from environment

    # Calculate loss
    loss_dict = criterion(outputs, actions_taken, units_selected, coords_taken, rewards)

    print(f"Total loss: {loss_dict['total_loss'].item():.4f}")
    print(f"Action loss: {loss_dict['action_loss'].item():.4f}")
    print(f"Unit loss: {loss_dict['unit_loss'].item():.4f}")
    print(f"Coordinate loss: {loss_dict['coord_loss'].item():.4f}")

if __name__ == "__main__":
    main()
