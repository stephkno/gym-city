"""
Random Agent using the proper gym interface for gym-city.

This version:
1. Uses the standard gym environment interface
2. Renders in real-time with GUI
3. Allows conditional action taking based on observations
4. Properly handles environment reset for new cities

Usage:
    python random_agent.py --map-width 20 --max-step 500
"""

import sys
import os
import time
import argparse

import gym
import numpy as np

# Import gym-city modules
import gym_city


def parse_args():
    parser = argparse.ArgumentParser(description='Random Agent for Gym-City')
    parser.add_argument('--map-width', type=int, default=20,
                        help='Width/height of the city map (default: 20)')
    parser.add_argument('--max-step', type=int, default=500,
                        help='Maximum steps per episode (default: 500)')
    parser.add_argument('--random-builds', type=str, default='true',
                        help='Whether to start with random builds (default: true)')
    parser.add_argument('--random-terrain', type=str, default='true',
                        help='Whether to use random terrain (default: true)')
    parser.add_argument('--render-delay', type=float, default=0.05,
                        help='Delay between steps for real-time rendering (default: 0.05)')
    parser.add_argument('--episode-delay', type=float, default=2.0,
                        help='Delay between episodes (default: 2.0)')
    return parser.parse_args()


def str_to_bool(val):
    """Convert string to boolean for argparse."""
    if isinstance(val, bool):
        return val
    if val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected')


def get_city_metrics(env):
    """Extract city metrics from the environment."""
    micro = env.unwrapped.micro
    engine = micro.engine
    
    metrics = {
        'res_pop': engine.resPop,
        'com_pop': engine.comPop,
        'ind_pop': engine.indPop,
        'total_pop': engine.totalPop,
        'traffic': micro.total_traffic,
        'num_plants': micro.map.num_plants,
        'funds': engine.totalFunds,
        'city_time': engine.cityTime,
        'city_month': engine.cityMonth,
        'year': (engine.cityTime // 48) + 1900,
        'mayor_rating': engine.cityYes,
    }
    return metrics


def print_metrics(env, step):
    """Print city metrics in a readable format using the new describe_city method."""
    # Use the new plain text description method
    description = env.describe_city(verbose=False)
    print(f"\n{description}")


def is_action_allowed(step, metrics, action_type):
    """
    Determine if an action should be taken based on current state.
    
    Args:
        step: Current step number
        metrics: Current city metrics
        action_type: Type of action to condition on
    
    Returns:
        bool: True if action should be taken
    """
    # Example conditions - customize these for your LLM agent
    
    # Condition 1: Take action every N steps
    if action_type == 'periodic' and step % 10 == 0:
        return True
    
    # Condition 2: Take action when population changes significantly
    if action_type == 'pop_change':
        if step > 0 and metrics['total_pop'] > 0:
            # Check if population changed significantly
            return True
    
    # Condition 3: Take action when funds are low
    if action_type == 'funds_low' and metrics['funds'] < 1000000:
        return True
    
    # Condition 4: Always take action (for random agent)
    if action_type == 'always':
        return True
    
    # Default: don't take action
    return False


def main():
    args = parse_args()
    
    # Convert string args to bool
    random_builds = str_to_bool(args.random_builds)
    random_terrain = str_to_bool(args.random_terrain)
    
    print(f"Creating Gym-City environment...")
    print(f"  Map width: {args.map_width}")
    print(f"  Max steps: {args.max_step}")
    print(f"  Random builds: {random_builds}")
    print(f"  Random terrain: {random_terrain}")
    print()
    
    # Note: MicropolisEnv doesn't set up action/observation spaces in __init__
    # It waits until setMapSize() is called. To use gym.make(), we need to
    # directly instantiate the class to avoid gym's PassiveEnvChecker validation.
    
    # Option 1: Direct instantiation (recommended for this environment)
    from gym_city.envs import MicropolisEnv
    env = MicropolisEnv(MAP_X=args.map_width, MAP_Y=args.map_width)
    
    # Configure the environment (this sets up the spaces)
    env.setMapSize(
        size=args.map_width,
        render_gui=True,
        empty_start=not random_terrain,  # empty_start=False means random terrain
        max_step=args.max_step,
        rank=0,
        random_builds=random_builds
    )
    
    # Now the environment is ready to use
    print(f'Action space: {env.action_space}')
    print(f'Observation space: {env.observation_space.shape}')
    
    # Reset to create initial city
    obs = env.reset()
    print(f"Environment reset. Initial observation shape: {obs.shape}")
    
    # Print city description
    print_metrics(env, step=0)
    
    # Action space info
    print(f"\nAction space: {env.action_space}")
    print(f"  Discrete with {env.action_space.n} possible actions")
    
    # For LLM agent: You can define custom conditions for when to take actions
    # Example: Only act every 10 steps, or only when population grows
    action_condition = 'always'  # Options: 'always', 'periodic', 'pop_change', 'funds_low'
    
    # Main loop
    step = 0
    episode = 0
    total_reward = 0
    episode_reward = 0
    
    try:
        while True:
            # Render and advance
            env.render()
            
            # Add delay for real-time viewing
            if args.render_delay > 0:
                time.sleep(args.render_delay)
            
            # Check if we should take an action
            # For LLM agent: this is where you'd insert your LLM decision logic
            # Get current metrics if needed for conditional logic
            current_metrics = get_city_metrics(env) if action_condition != 'always' else None
            take_action = is_action_allowed(step, current_metrics, action_condition)
            
            if take_action:
                # Sample random action
                action = env.action_space.sample()
                
                # Take the action
                obs, reward, done, info = env.step(action)
                total_reward += reward
                episode_reward += reward
                
                # Print city state
                print_metrics(env, step)
                
                # Optional: Print observation stats
                if step % 50 == 0:
                    print(f"  Observation stats - min: {obs.min():.3f}, max: {obs.max():.3f}, mean: {obs.mean():.3f}")
            else:
                # Skip action - engine still advances simulation
                # For LLM agent: you might want to query the environment state
                # without taking an action
                if step % 10 == 0:
                    print(f"\nStep {step} (skipping action)")
            
            step += 1

            print(env.zone_grid_to_text(False))
            
            # Check if episode is done
            if done:
                print(f"\n{'='*60}")
                print(f"EPISODE {episode} COMPLETE")
                print(f"  Total steps: {step}")
                print(f"  Episode reward: {episode_reward:.2f}")
                print(f"  Total reward: {total_reward:.2f}")
                print(f"{'='*60}\n")
                
                # Wait before next episode
                time.sleep(args.episode_delay)
                
                # Reset for new city
                obs = env.reset()
                episode += 1
                episode_reward = 0
                step = 0
                
                print(f"Environment reset for new city (Episode {episode})")
                #print_metrics(env, step=0)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Closing environment...")
        env.close()
        print("Done.")


if __name__ == '__main__':
    main()
