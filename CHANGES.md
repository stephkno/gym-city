# Gym-City Random Agent Updates

## Summary
Repurposed the random_agent.py to use the proper gym interface instead of directly interacting with the GTK GUI engine.

## Changes Made

### 1. Backup Created
- `random_agent.bak` - Contains the original code that directly manipulated the GTK engine

### 2. New random_agent.py Features

The new implementation:

1. **Uses the gym environment interface** - Directly instantiates `MicropolisEnv` class instead of using the GTK `main.train()` function

2. **Proper environment initialization** - Calls `setMapSize()` after instantiation to properly set up action/observation spaces

3. **Real-time rendering** - Uses `env.render()` which processes GTK events for display updates

4. **Conditional action taking** - Includes `is_action_allowed()` function that allows you to control when actions are taken based on environment state

5. **Plain text city descriptions** - Uses the new `describe_city()` method for readable state output

6. **Episode handling** - Properly resets the environment for new cities when episodes complete

### 3. Environment Changes

#### New `describe_city()` Method

Added a new method to `MicropolisEnv` in `gym_city/envs/env.py`:

```python
def describe_city(self, verbose=False):
    """Returns a plain text description of the current city state."""
```

This method provides a human-readable string description including:
- Population breakdown (total, residential, commercial, industrial)
- Infrastructure (traffic, power plants, funds, mayor rating)
- Zone distribution (tile counts and percentages)
- Optional verbose stats (step, episode, reward, time)

**Example output:**
```
============================================================
CITY STATE DESCRIPTION
============================================================

POPULATION:
  Total: 240
  Residential: 150
  Commercial: 50
  Industrial: 40

INFRASTRUCTURE:
  Traffic: 1200
  Power Plants: 2
  Funds: $1,850,000
  Mayor Rating: 65/100

ZONE DISTRIBUTION:
  Residential: 85 tiles (21.25%)
  Commercial: 35 tiles (8.75%)
  Industrial: 28 tiles (7.0%)
  Road: 120 tiles (30.0%)
  Wire: 45 tiles (11.25%)
  Land: 87 tiles (21.75%)

METRICS:
  Step: 150
  Episode: 3
  Current Reward: 125.50
  City Time: 7213 ticks
  City Month: 13
  Year: 1915
============================================================
```

### 4. Key Design Decisions

**Why direct instantiation instead of `gym.make()`?**

The `MicropolisEnv` class doesn't set up action/observation spaces in `__init__()` - it waits until `setMapSize()` is called. The newer version of Gym includes a `PassiveEnvChecker` wrapper that validates environments immediately upon `make()`, which fails because spaces aren't initialized yet.

**Solution:** Directly instantiate `MicropolisEnv` class, then call `setMapSize()` to initialize everything properly.

### 4. LLM Agent Integration Points

The code is designed for easy LLM integration:

```python
# In main loop, find where actions are decided:
take_action = is_action_allowed(step, metrics, action_condition)

if take_action:
    # Your LLM decision logic here
    # action = llm_policy(obs, metrics)
    action = env.action_space.sample()  # Currently random
```

The `is_action_allowed()` function shows how to condition actions on:
- Periodic intervals
- Population changes
- Low funds
- Any custom condition

### 5. Usage

```bash
# Basic usage with default settings
python random_agent.py

# Custom settings
python random_agent.py \
  --map-width 20 \
  --max-step 500 \
  --render-delay 0.05 \
  --episode-delay 2.0 \
  --random-builds true \
  --random-terrain true
```

## Environment Reset Process

To create a new city:

1. **In the training loop:** When `done=True`, call `env.reset()`
2. **The `reset()` method in `env.py`**:
   - Clears the map (`micro.clearMap()`)
   - Generates new terrain if `empty_start=False`
   - Resets step counter
   - Sets up power puzzle or random builds if enabled
   - Advances simulation one tick
   - Computes initial city metrics
   - Resets funds
   - Returns initial observation

The key is that `reset()` creates a completely new city state each time.

## Key Methods for LLM Agents

### Getting Plain Text City Descriptions
```python
# Get a full plain text description of the city state
city_description = env.describe_city(verbose=False)
print(city_description)
```

The `describe_city()` method returns a formatted string with:
- Population breakdown (total, residential, commercial, industrial)
- Infrastructure (traffic, power plants, funds, mayor rating)
- Zone distribution (how many tiles of each zone type)
- Optional verbose stats (step, episode, current reward, time)

### Getting City Metrics as Dictionary
```python
metrics = get_city_metrics(env)
# Returns: res_pop, com_pop, ind_pop, total_pop, traffic, 
#          num_plants, funds, city_time, city_month, year, mayor_rating
```

### Conditional Action Taking
```python
take_action = is_action_allowed(step, metrics, condition_type)
# Options: 'always', 'periodic', 'pop_change', 'funds_low'
```

### Observation Format
```python
obs.shape = (num_obs_channels, MAP_X, MAP_Y)
# Includes: zone maps, density maps, scalar layers, static builds
```

## Notes

- The render loop uses `time.sleep()` for real-time viewing - adjust `--render-delay` as needed
- Actions are skipped based on `is_action_allowed()` but simulation still advances
- Funds are auto-replenished each step to prevent bankruptcy termination
- The environment runs until `max_step` is reached or funds go negative
