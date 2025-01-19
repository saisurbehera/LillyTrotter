import torch
import torch.nn as nn

class SpatialEncoder(nn.Module):
    def __init__(self, width=24, height=24):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # 3 channels: energy, tile_type, sensor_mask
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(64 * (width//2) * (height//2), 256)

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x.view(x.size(0), -1))

class UnitProcessor(nn.Module):
    def __init__(self, max_units=10):
        super().__init__()
        self.max_units = max_units
        self.fc1 = nn.Linear(3, 64)  # position (2) + energy (1)
        self.fc2 = nn.Linear(64, 128)
        
    def forward(self, positions, energy, mask):
        # Reshape it to add it in  the last deminsion
        energy = energy.unsqueeze(-1) 
        # Combine position and energy
        features = torch.cat([positions, energy], dim=-1)  # Shape: (batch, teams, units, 3)
        
        # Process each unit
        x = torch.relu(self.fc1(features))
        x = torch.relu(self.fc2(x))
        
        # Apply mask
        x = x * mask.unsqueeze(-1)
        
        # Pool over units dimension
        return torch.max(x, dim=2)[0]  # Shape: (batch, teams, 128)

class RelicProcessor(nn.Module):
    def __init__(self, max_relics=10):
        super().__init__()
        self.fc1 = nn.Linear(2, 64)  # position only
        self.fc2 = nn.Linear(64, 128)
        
    def forward(self, positions, mask):
        x = torch.relu(self.fc1(positions))
        x = torch.relu(self.fc2(x))
        
        # Apply mask
        x = x * mask.unsqueeze(-1)
        
        # Pool over relics
        return torch.max(x, dim=1)[0]

class LuxAIObservationEncoder(nn.Module):
    def __init__(self, width=24, height=24, max_units=10, max_relics=10):
        super().__init__()
        self.spatial_encoder = SpatialEncoder(width, height)
        self.unit_processor = UnitProcessor(max_units)
        self.relic_processor = RelicProcessor(max_relics)

        # Calculate total embedding dimension
        spatial_dim = 256
        units_dim = 128 * 2  # For 2 teams
        relic_dim = 128
        total_dim = spatial_dim + units_dim + relic_dim
        
        self.output_layer = nn.Linear(total_dim, 512)

    def forward(self, obs):
        # Process spatial features
        spatial_features = torch.cat([
            obs['map_features']['energy'].unsqueeze(1),
            obs['map_features']['tile_type'].unsqueeze(1),
            obs['sensor_mask'].unsqueeze(1)
        ], dim=1)
        spatial_embed = self.spatial_encoder(spatial_features)

        # Process units
        unit_embed = self.unit_processor(
            obs['units']['position'],
            obs['units']['energy'],
            obs['units_mask']
        )
        unit_embed = unit_embed.reshape(unit_embed.size(0), -1)  # Flatten teams dimension

        # Process relics
        relic_embed = self.relic_processor(
            obs['relic_nodes'],
            obs['relic_nodes_mask']
        )

        # Combine all embeddings
        combined = torch.cat([
            spatial_embed,
            unit_embed,
            relic_embed
        ], dim=1)

        return self.output_layer(combined)

class GameStateProcessor:
    def process_observation(self, obs):
        """Convert raw observation dict into tensor format"""
        return {
            'map_features': {
                'energy': torch.FloatTensor(obs['obs']['map_features']['energy']),
                'tile_type': torch.FloatTensor(obs['obs']['map_features']['tile_type'])
            },
            'sensor_mask': torch.FloatTensor(obs['obs']['sensor_mask']),
            'units': {
                'position': torch.FloatTensor(obs['obs']['units']['position']),
                'energy': torch.FloatTensor(obs['obs']['units']['energy'])
            },
            'units_mask': torch.FloatTensor(obs['obs']['units_mask']),
            'relic_nodes': torch.FloatTensor(obs['obs']['relic_nodes']),
            'relic_nodes_mask': torch.FloatTensor(obs['obs']['relic_nodes_mask'])
        }
        
