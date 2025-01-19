import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseNetwork(nn.Module):
    """Shared base network for processing state input"""
    def __init__(self, input_dim=64, lstm_hidden_size=256, embedding_dim=64):
        super(BaseNetwork, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.embedding_dim = embedding_dim
        
        # LSTM for processing state
        self.lstm = nn.LSTM(
            input_size=input_dim,
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
        lstm_out, _ = self.lstm(state)
        lstm_final = lstm_out[:, -1, :]
        return lstm_final

class PolicyNetwork(nn.Module):
    """Policy network with attention-based unit targeting"""
    def __init__(self, base_network, num_units=16, num_actions=2):
        super(PolicyNetwork, self).__init__()
        self.base = base_network
        self.num_units = num_units
        self.num_actions = num_actions  # 0 = move, 1 = attack
        
        # Action selection layers
        self.action_fc = nn.Linear(base_network.lstm_hidden_size, base_network.embedding_dim)
        self.action_embedding = nn.Embedding(num_actions, base_network.embedding_dim)
        
        # Unit targeting layers
        self.unit_fc = nn.Linear(base_network.lstm_hidden_size, base_network.embedding_dim)
        self.unit_embedding = nn.Embedding(num_units, base_network.embedding_dim)
        
        # Learned per-action mask for unit targeting
        self.action_unit_mask = nn.Parameter(torch.randn(num_actions, num_units))
        
        # Coordinate prediction
        self.coord_fc = nn.Linear(base_network.lstm_hidden_size, 2)

    def forward(self, state, available_actions, available_units):
        """
        Forward pass implementing attention-based unit targeting
        Args:
            state: Input state tensor (batch_size, seq_len, input_dim)
            available_actions: Boolean mask of available actions (batch_size, num_actions)
            available_units: Boolean mask of available units (batch_size, num_units)
        Returns:
            Dictionary containing action distributions and values
        """
        lstm_final = self.base(state)
        batch_size = state.shape[0]
        
        # 1. Action Selection
        action_embeds = self.action_embedding(torch.arange(self.num_actions).to(state.device))
        action_query = self.action_fc(lstm_final)
        action_scores = torch.matmul(action_query, action_embeds.t())
        action_scores = action_scores.masked_fill(~available_actions, float('-inf'))
        action_probs = F.softmax(action_scores, dim=-1)  # [batch_size, num_actions]
        
        # 2. Unit Targeting with attention
        unit_embeds = self.unit_embedding(torch.arange(self.num_units).to(state.device))
        unit_query = self.unit_fc(lstm_final)  # [batch_size, embedding_dim]
        
        # Calculate attention scores for each action-unit combination
        unit_scores = torch.matmul(unit_query, unit_embeds.t())  # [batch_size, num_units]
        
        # Apply learned per-action mask
        action_unit_masks = self.action_unit_mask.unsqueeze(0)  # [1, num_actions, num_units]
        unit_scores = unit_scores.unsqueeze(1)  # [batch_size, 1, num_units]
        
        # Combine with action mask and available units mask
        masked_unit_scores = unit_scores + action_unit_masks
        masked_unit_scores = masked_unit_scores.masked_fill(~available_units.unsqueeze(1), float('-inf'))
        unit_probs = F.softmax(masked_unit_scores, dim=-1)  # [batch_size, num_actions, num_units]
        
        # 3. Coordinate prediction
        coordinates = self.coord_fc(lstm_final)  # [batch_size, 2]
        
        return {
            'action_probs': action_probs,    # [batch_size, num_actions]
            'unit_probs': unit_probs,        # [batch_size, num_actions, num_units]
            'coordinates': coordinates        # [batch_size, 2]
        }

    def sample_action(self, outputs):
        """Sample action tuple (action_type, target_unit, coordinates)"""
        batch_size = outputs['action_probs'].shape[0]
        
        # Sample action type first (with bias towards move actions during inference)
        if self.training:
            actions = torch.multinomial(outputs['action_probs'], 1)  # [batch_size, 1]
        else:
            # Bias towards move actions during inference
            probs = outputs['action_probs'].clone()
            probs[:, 1] *= 0.1  # Reduce attack probability
            probs = F.normalize(probs, p=1, dim=1)
            actions = torch.argmax(probs, dim=-1, keepdim=True)
            
        # Get unit probabilities for the sampled actions
        batch_indices = torch.arange(batch_size)
        unit_probs = outputs['unit_probs'][batch_indices, actions.squeeze()]
        
        # Sample target unit based on the action-specific probabilities
        if self.training:
            units = torch.multinomial(unit_probs, 1)
        else:
            units = torch.argmax(unit_probs, dim=-1, keepdim=True)
            
        return actions, units, outputs['coordinates']

class ValueNetwork(nn.Module):
    """Value network for state value estimation"""
    def __init__(self, base_network):
        super(ValueNetwork, self).__init__()
        self.base = base_network
        
        # Value head
        self.value_fc1 = nn.Linear(base_network.lstm_hidden_size, 128)
        self.value_fc2 = nn.Linear(128, 1)
    
    def forward(self, state):
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
    
    def forward(self, outputs, actions_taken, units_selected, coords_taken):
        batch_size = actions_taken.shape[0]
        
        # Action type loss
        action_loss = -self.action_weight * torch.mean(
            torch.log(outputs['action_probs'].gather(1, actions_taken))
        )
        
        # Unit selection loss
        batch_indices = torch.arange(batch_size)
        selected_unit_probs = outputs['unit_probs'][batch_indices, actions_taken.squeeze()]
        unit_loss = -self.unit_weight * torch.mean(
            torch.log(selected_unit_probs.gather(1, units_selected))
        )
        
        # Coordinate loss
        coord_loss = self.coord_weight * F.mse_loss(
            outputs['coordinates'], coords_taken
        )
        
        return {
            'total_loss': action_loss + unit_loss + coord_loss,
            'action_loss': action_loss,
            'unit_loss': unit_loss,
            'coord_loss': coord_loss
        }

def main():
    # Example parameters
    batch_size = 3
    seq_len = 100
    input_dim = 64
    num_units = 16
    
    # Create networks
    base_net = BaseNetwork(input_dim=input_dim)
    policy_net = PolicyNetwork(base_net, num_units=num_units)
    value_net = ValueNetwork(base_net)
    
    # Create loss function
    bc_criterion = BehaviorCloningLoss()
    
    # Sample inputs
    state = torch.randn(batch_size, seq_len, input_dim)
    available_actions = torch.ones(batch_size, 2, dtype=torch.bool)
    available_units = torch.ones(batch_size, num_units, dtype=torch.bool)
    
    # Forward pass
    policy_outputs = policy_net(state, available_actions, available_units)
    value = value_net(state)
    
    # Sample some actions
    actions_taken, units_selected, coords_taken = policy_net.sample_action(policy_outputs)
    
    print("\nNetwork Outputs:")
    print(f"Action probs shape: {policy_outputs['action_probs'].shape}")
    print(f"Unit probs shape: {policy_outputs['unit_probs'].shape}")
    print(f"Coordinates shape: {policy_outputs['coordinates'].shape}")
    
    print("\nSampled Actions:")
    print(f"actions_taken: {actions_taken}")
    print(f"units_selected: {units_selected}")
    print(f"coords_taken: {coords_taken}")
    
    # Calculate loss
    loss_dict = bc_criterion(policy_outputs, actions_taken, units_selected, coords_taken)
    
    print("\nLosses:")
    for key, value in loss_dict.items():
        print(f"{key}: {value.item():.4f}")

if __name__ == "__main__":
    main()