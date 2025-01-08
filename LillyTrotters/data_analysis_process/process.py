import json
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
from glob import glob 

@dataclass
class GameStats:
    round_won: bool
    relic_points: int
    total_energy: int
    units_killed: int
    relic_tiles_found: float
    area_explored: float

def parse_observation(obs):
    """
    Convert raw observation dict into tensor format required by LuxAIObservationEncoder
    
    Args:
        raw_obs (dict): Raw observation dictionary from the environment
        
    Returns:
        dict: Processed observation with tensors
    """
    
    return {
        'map_features': {
            'energy': torch.FloatTensor(obs['map_features']['energy']).unsqueeze(0),
            'tile_type': torch.FloatTensor(obs['map_features']['tile_type']).unsqueeze(0)
        },
        'sensor_mask': torch.FloatTensor(obs['sensor_mask']).unsqueeze(0),
        'units': {
            'position': torch.FloatTensor(obs['units']['position']).unsqueeze(0),
            'energy': torch.FloatTensor(obs['units']['energy']).unsqueeze(0)
        },
        'units_mask': torch.FloatTensor(obs['units_mask']).unsqueeze(0),
        'relic_nodes': torch.FloatTensor(obs['relic_nodes']).unsqueeze(0),
        'relic_nodes_mask': torch.FloatTensor(obs['relic_nodes_mask']).unsqueeze(0)
    }
class LuxGameParser:
    def __init__(self, map_size=24, max_units=10):
        self.map_size = map_size
        self.max_units = max_units
    
    def parse_game_file(self, file_path: str) -> List[GameStats]:
        with open(file_path, 'r') as f:
            game_data = json.load(f)
        env_stats = []
        game_stats = []
        for step in game_data['steps']:
            for i, each_player in enumerate(step):
                obs = each_player["observation"]["obs"]
                obs = json.loads(obs)
                env_stats.append(parse_observation(obs))
                
                total_energy = obs['units']['energy'][i]
                total_energy = sum(num for num in total_energy if num != -1)
                
                game_stats.append(GameStats(
                    round_won=obs['team_wins'][i] ,
                    relic_points=obs['team_points'][i],
                    total_energy=total_energy,
                    relic_tiles_found=np.mean(obs['relic_nodes_mask']),
                    area_explored=np.mean(obs['sensor_mask'])
                ))
        return env_stats, game_stats
    
    def _process_observation(self, obs) -> GameStats:
        return GameStats(
            match_won=obs['team_wins'][0] > 0,  # Assuming player 0
            round_won=obs['team_points'][0] > obs['team_points'][1],
            relic_points=obs['team_points'][0],
            total_energy=np.sum(obs['units']['energy'][0]),
            units_killed=np.sum(np.logical_not(obs['units_mask'][1])),
            relic_tiles_found=np.mean(obs['relic_node_configs']),
            area_explored=np.mean(obs['vision_power_map'] > 0),
            replay_observation=obs
        )

class GameDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, split: str = 'train', val_ratio: float = 0.2, seed: int = 42):
        self.parser = LuxGameParser()
        
        files = glob(data_dir+"/*/*.json") # Sort for reproducibility
        print(files)
        np.random.seed(seed)
        split_idx = int(len(files) * (1 - val_ratio))
        
        # Split files deterministically
        self.files = files[:split_idx] if split == 'train' else files[split_idx:]
        
        # Pre-load all game stats
        # self.games = []
        # for file in self.files:
        #     self.games.extend(self.parser.parse_game_file(str(file)))
    
    def __len__(self):
        return len(self.games)
    
    def __getitem__(self, idx):
        print(self.files[idx])
        env_stats, game_stats_all = luxParse.parse_game_file( self.files[idx])
        print(game_stats_all[0])
        
        labels = torch.tensor([(
            game_stats.round_won,
            game_stats.relic_points,
            game_stats.total_energy,
            game_stats.units_killed,
            game_stats.relic_tiles_found,
            game_stats.area_explored) for game_stats in game_stats_all])
        
        return env_stats, labels

def create_dataloaders(data_dir: str, batch_size: int = 32, val_ratio: float = 0.2, 
                      num_workers: int = 4, seed: int = 42):
    train_dataset = GameDataset(data_dir, 'train', val_ratio, seed)
    val_dataset = GameDataset(data_dir, 'val', val_ratio, seed)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader