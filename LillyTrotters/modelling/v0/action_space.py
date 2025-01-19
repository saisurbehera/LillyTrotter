
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseNetwork(nn.Module):
    """Shared base network for processing state input"""
    def __init__(self, lstm_hidden_size=256, embedding_dim=64):
        super(BaseNetwork, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.embedding_dim = embedding_dim
        
        # LSTM for processing state
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_size,
            batch_first=True
        )

    def forward(self, state):
        """
        Process state through LSTM
        Args:
            state: Input state tensor (batch_size, seq_len, input_dim)
        Returns:
            lstm_final: Final LSTM hidden state (batch_size, lstm_hidden_size)
        """
        lstm_out, _ = self.lstm(state)  # [batch_size, seq_len, lstm_hidden_size]
        lstm_final = lstm_out[:, -1, :]  # [batch_size, lstm_hidden_size]
        return lstm_final

class PolicyNetwork(nn.Module):
    """Policy network for action selection"""
    def __init__(self, base_network, action_dim=2, unit_dim=16):
        super(PolicyNetwork, self).__init__()
        self.base = base_network
        self.action_dim = action_dim
        self.unit_dim = unit_dim
        
        # Action ID path
        self.action_embedding = nn.Embedding(action_dim, base_network.embedding_dim)
        self.action_fc = nn.Linear(base_network.lstm_hidden_size, base_network.embedding_dim)
        
        # Target unit path
        self.unit_embedding = nn.Embedding(self.unit_dim, base_network.embedding_dim)
        self.unit_fc = nn.Linear(base_network.lstm_hidden_size, base_network.embedding_dim)
        
        # Coordinate prediction
        self.coord_fc = nn.Linear(base_network.lstm_hidden_size, self.unit_dim * 2)

    def forward(self, state, available_actions, available_units):
        """
        Args:
            state: Input state tensor (batch_size, seq_len, input_dim)
            available_actions: Boolean mask of available actions (batch_size, action_dim)
            available_units: Boolean mask of available units (batch_size, unit_dim)
        Returns:
            Dictionary containing action distributions for each unit
        """
        lstm_final = self.base(state)
        
        # Target unit selection - now returns probabilities for EACH unit
        unit_embeds = self.unit_embedding(torch.arange(self.unit_dim).to(state.device))
        unit_query = self.unit_fc(lstm_final)
        unit_scores = torch.matmul(unit_query, unit_embeds.t())
        unit_scores = unit_scores.masked_fill(~available_units, float('-inf'))
        # Instead of softmax, use sigmoid to allow multiple units to be selected
        unit_probs = torch.sigmoid(unit_scores)
        
        # Action ID selection for each available unit
        action_embeds = self.action_embedding(torch.arange(self.action_dim).to(state.device))
        action_query = self.action_fc(lstm_final)
        action_scores = torch.matmul(action_query, action_embeds.t())
        action_scores = action_scores.masked_fill(~available_actions, float('-inf'))
        
        # Get action probabilities for each unit
        action_probs = F.softmax(action_scores.unsqueeze(1).expand(-1, self.unit_dim, -1), dim=-1)
        
        # Coordinate prediction for each unit
        coords_all = self.coord_fc(lstm_final)
        coordinates = coords_all.view(-1, self.unit_dim, 2)
        
        return {
            'action_probs': action_probs,  # [batch_size, unit_dim, action_dim]
            'unit_probs': unit_probs,      # [batch_size, unit_dim]
            'coordinates': coordinates,     # [batch_size, unit_dim, 2]
        }

    def sample_actions(self, outputs):
        """Sample actions for each selected unit"""
        unit_mask = (outputs['unit_probs'] > 0.5)  # Binary selection of units
        
        if self.training:
            actions = torch.multinomial(outputs['action_probs'][unit_mask], 1)
        else:
            actions = torch.argmax(outputs['action_probs'][unit_mask], dim=-1, keepdim=True)
        
        return {
            'selected_units': unit_mask,
            'actions': actions,
            'coordinates': outputs['coordinates'][unit_mask]
        }
        
class ValueNetwork(nn.Module):
    """Value network for state value estimation"""
    def __init__(self, base_network):
        super(ValueNetwork, self).__init__()
        self.base = base_network
        
        # Value head
        self.value_fc1 = nn.Linear(base_network.lstm_hidden_size, 128)
        self.value_fc2 = nn.Linear(128, 1)
    
    def forward(self, state):
        """
        Args:
            state: Input state tensor (batch_size, seq_len, input_dim)
        Returns:
            value: Predicted state value (batch_size, 1)
        """
        lstm_final = self.base(state)
        hidden = F.relu(self.value_fc1(lstm_final))
        value = self.value_fc2(hidden)
        return value

class BehaviorCloningLoss(nn.Module):
    """Loss function for behavior cloning phase"""
    def __init__(self, action_weight=1.0, unit_weight=1.0, coord_weight=1.0):
        super(BehaviorCloningLoss, self).__init__()
        self.action_weight = action_weight
        self.unit_weight = unit_weight
        self.coord_weight = coord_weight

    def forward(self, outputs, target_units, target_actions, target_coords):
        """
        Args:
            outputs: Dictionary from policy network
            target_units: Binary mask of which units should be selected [batch_size, unit_dim]
            target_actions: Actions for selected units [num_selected_units]
            target_coords: Coordinates for selected units [num_selected_units, 2]
        """
        # Binary cross entropy for unit selection
        unit_loss = F.binary_cross_entropy_with_logits(
            outputs['unit_probs'], target_units.float()
        )
        
        # Cross entropy loss for actions of selected units
        selected_mask = target_units.bool()
        action_loss = -torch.mean(
            torch.log(outputs['action_probs'][selected_mask].gather(1, target_actions))
        )
        
        # MSE loss for coordinates of selected units
        coord_loss = F.mse_loss(
            outputs['coordinates'][selected_mask], target_coords
        )
        
        return {
            'total_loss': unit_loss + action_loss + coord_loss,
            'unit_loss': unit_loss,
            'action_loss': action_loss,
            'coord_loss': coord_loss
        }

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# Example usage
def main():
    batch_size = 16
    seq_len = 500
    input_dim = 64
    
    # Create networks
    base_net = BaseNetwork()
    policy_net = PolicyNetwork(base_net)
    value_net = ValueNetwork(base_net)
    
    # Create loss function for behavior cloning
    bc_criterion = BehaviorCloningLoss(action_weight=1.0, unit_weight=0.8, coord_weight=0.5)
    
    # Sample inputs
    state = torch.randn(batch_size, seq_len, input_dim)
    available_actions = torch.ones(batch_size, 2, dtype=torch.bool)
    available_units = torch.ones(batch_size, 16, dtype=torch.bool)
    
    # Forward passes
    policy_outputs = policy_net(state, available_actions, available_units)
    state_values = value_net(state)
    
    # Create sample target data - now with multiple units per batch
    target_units = torch.zeros(batch_size, 16, dtype=torch.bool)
    # Randomly select 2-4 units per batch
    for i in range(batch_size):
        num_units = torch.randint(2, 5, (1,)).item()
        selected_units = torch.randperm(16)[:num_units]
        target_units[i, selected_units] = True
    
    # Generate actions and coordinates only for selected units
    num_selected = target_units.sum().item()
    target_actions = torch.randint(0, 2, (num_selected, 1))
    target_coords = torch.randn(num_selected, 2)
    
    # Calculate behavior cloning loss
    bc_loss_dict = bc_criterion(
        policy_outputs,
        target_units,
        target_actions,
        target_coords
    )
    
    # Sample actions for inference
    sampled_actions = policy_net.sample_actions(policy_outputs)
    
    print(f"Parameters in base network: {count_parameters(base_net)}")
    print(f"Parameters in policy network: {count_parameters(policy_net)}")
    print(f"Parameters in value network: {count_parameters(value_net)}")
    
    print(f"\nBehavior Cloning Losses:")
    print(f"Total loss: {bc_loss_dict['total_loss'].item():.4f}")
    print(f"Action loss: {bc_loss_dict['action_loss'].item():.4f}")
    print(f"Unit loss: {bc_loss_dict['unit_loss'].item():.4f}")
    print(f"Coordinate loss: {bc_loss_dict['coord_loss'].item():.4f}")
    
    print(f"\nNetwork Outputs:")
    print(f"Action probs shape: {policy_outputs['action_probs'].shape}")
    print(f"Unit probs shape: {policy_outputs['unit_probs'].shape}")
    print(f"Coordinates shape: {policy_outputs['coordinates'].shape}")
    
    print(f"\nSampled Actions:")
    print(f"Number of units selected: {sampled_actions['selected_units'].sum().item()}")
    print(f"Actions shape: {sampled_actions['actions'].shape}")
    print(f"Coordinates shape: {sampled_actions['coordinates'].shape}")
    
    print(f"\nState values shape: {state_values.shape}")
    
    
if __name__ == "__main__":
    main()