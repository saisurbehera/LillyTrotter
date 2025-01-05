"""
In this file we will be the layers for handling the observation space.

We are only encoding at time = t, so the final embedding is of size (batch, t, final_dim)

Since, I will be using this in LSTM network to combine spaces together,
we will let the LSTM take care of the state-space dimension.
"""

import torch
import torch.nn as nn

class SpatialEncoder(nn.Module):
    """Processes spatial information like map features"""
    def __init__(self, map_size=24):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),  # 4 channels: energy, tile_type, vision_power, units_mask
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(64 * (map_size//2) * (map_size//2), 256)

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x.view(x.size(0), -1))

class UnitSetProcessor(nn.Module):
    """Processes variable-sized sets of units using Process Set approach"""
    def __init__(self, input_dim=3):  # position (2) + energy (1)
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)

    def forward(self, x):
        # x shape: (batch, num_units, features)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # Max pooling over units
        return torch.max(x, dim=1)[0]

class NodeProcessor(nn.Module):
    """Processes energy and relic nodes"""
    def __init__(self, node_type='energy'):
        super().__init__()
        input_dim = 2  # position (x,y)
        if node_type == 'relic':
            input_dim += 25  # 5x5 configuration mask flattened

        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.max(x, dim=1)[0]  # max pool over nodes

class LuxAIObservationEncoder(nn.Module):
    def __init__(self, map_size=24):
        super().__init__()
        self.spatial_encoder = SpatialEncoder(map_size)
        self.unit_processor = UnitSetProcessor()
        self.energy_node_processor = NodeProcessor('energy')
        self.relic_node_processor = NodeProcessor('relic')

        # Final embedding dimension
        total_dim = 256 + 128 + 128 + 128  # spatial + units + energy_nodes + relic_nodes
        self.output_layer = nn.Linear(total_dim, 512)

    def forward(self, obs):
        # Process spatial features
        spatial_features = torch.cat([
            obs['map_features']['energy'],
            obs['map_features']['tile_type'],
            obs['vision_power_map'],
            obs['units_mask']
        ], dim=1)
        spatial_embed = self.spatial_encoder(spatial_features)

        # Process units
        unit_positions = obs['units']['position']
        unit_energy = obs['units']['energy']
        unit_features = torch.cat([unit_positions, unit_energy.unsqueeze(-1)], dim=-1)
        unit_embed = self.unit_processor(unit_features)

        # Process nodes
        energy_node_embed = self.energy_node_processor(obs['energy_nodes'])

        # Process relic nodes with their configs
        relic_features = torch.cat([
            obs['relic_nodes'],
            obs['relic_node_configs'].view(obs['relic_nodes'].size(0), -1)
        ], dim=-1)
        relic_node_embed = self.relic_node_processor(relic_features)

        # Combine all embeddings
        combined = torch.cat([
            spatial_embed,
            unit_embed,
            energy_node_embed,
            relic_node_embed
        ], dim=1)

        return self.output_layer(combined)

class GameStateProcessor:
    """Processes raw observations into format suitable for the encoder"""
    def __init__(self, map_size=24):
        self.map_size = map_size

    def process_observation(self, obs):
        """Convert observation dict into tensor format"""
        # Example processing - would need to be adapted to actual observation format
        return {
            'map_features': {
                'energy': torch.tensor(obs['map_features']['energy']),
                'tile_type': torch.tensor(obs['map_features']['tile_type'])
            },
            'vision_power_map': torch.tensor(obs['vision_power_map']),
            'units_mask': torch.tensor(obs['units_mask']),
            'units': {
                'position': torch.tensor(obs['units']['position']),
                'energy': torch.tensor(obs['units']['energy'])
            },
            'energy_nodes': torch.tensor(obs['energy_nodes']),
            'relic_nodes': torch.tensor(obs['relic_nodes']),
            'relic_node_configs': torch.tensor(obs['relic_node_configs'])
        }
