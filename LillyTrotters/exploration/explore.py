GAME_PARAMS = {
    'map_size': 24,  # 24x24 grid
    'max_steps': 100,  # Each game lasts 100 steps
    'max_units': 12,   # Typically 3-4 units per team
    'map_features': 5  # empty, asteroid, nebula, energy, relic
}

# Minimum Dataset Size (100 games):
min_dataset = {
    'states_per_game': 100,  # steps
    'num_games': 100,
    'total_states': 100 * 100,  # 10,000 states
    'states_per_unit': 10000 * 4,  # ~40,000 unit states
}

# Decent Dataset Size (500 games):
decent_dataset = {
    'states_per_game': 100,
    'num_games': 500,
    'total_states': 500 * 100,  # 50,000 states
    'states_per_unit': 50000 * 4,  # ~200,000 unit states
}

def explain_dataset_size(num_games=500):
    # Game complexity factors
    unique_scenarios = {
        'early_game': 20,    # First 20 steps
        'mid_game': 50,      # Middle 50 steps
        'late_game': 30,     # Last 30 steps
        'unit_counts': 4,    # Average units per team
        'map_variations': 24 * 24,  # Different positions
        'resource_states': 5  # Different resource levels
    }

    # Minimal states needed for learning
    min_samples_per_scenario = 10

    # Calculate required states
    required_states = (
        unique_scenarios['early_game'] +
        unique_scenarios['mid_game'] +
        unique_scenarios['late_game']
    ) * unique_scenarios['unit_counts'] * min_samples_per_scenario

    # States from num_games
    total_states = num_games * 100  # steps per game

    return {
        'required_states': required_states,
        'provided_states': total_states,
        'coverage_ratio': total_states / required_states
    }

# Example
coverage = explain_dataset_size(500)
print(f"Required states: {coverage['required_states']}")
print(f"Provided states: {coverage['provided_states']}")
print(f"Coverage ratio: {coverage['coverage_ratio']:.2f}x")

def calculate_scenario_coverage(num_games):
    # Key game scenarios
    scenarios = {
        'unit_positions': 24 * 24,  # Possible positions
        'resource_locations': 24 * 24,
        'unit_interactions': 4 * 4,  # Unit vs unit scenarios
        'energy_levels': 5,  # Different energy states
        'action_types': 6    # Move directions + sap
    }

    # States per game
    states_per_game = 100
    total_states = num_games * states_per_game

    # Required samples for each scenario type
    min_samples = 10  # Want to see each scenario multiple times

    total_scenarios = (
        scenarios['unit_positions'] *
        scenarios['unit_interactions'] *
        scenarios['energy_levels'] *
        scenarios['action_types']
    )

    return {
        'total_scenarios': total_scenarios,
        'covered_scenarios': total_states,
        'coverage_percentage': (total_states / total_scenarios) * 100
    }

# Different dataset sizes
for games in [100, 500,1000, 2000, 5000]:
    coverage = calculate_scenario_coverage(games)
    print(f"\nGames: {games}")
    print(f"Total possible scenarios: {coverage['total_scenarios']:,}")
    print(f"Covered scenarios: {coverage['covered_scenarios']:,}")
    print(f"Coverage: {coverage['coverage_percentage']:.2f}%")
