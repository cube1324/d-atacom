# Handling Long-Term Safety and Uncertainty in Safe Reinforcement Learning

Code for the [paper](www.google.com) Handling Long-Term Safety and Uncertainty in Safe Reinforcement Learning. 

<p align="center">
<img src=figs/air_hockey.gif height="280">
<img src=figs/fvf.gif height="280" width="280">
</p>

## Usage
We use the [uv package manager](https://docs.astral.sh/uv/getting-started/installation/). If you don't want to use `uv` we provide a `requirements.txt` for manual installation. 
```python
git clone https://github.com/cube1324/d-atacom.git
cd d-atacom/cremini_rl
```
To train D-ATACOM on the Planar Air Hockey environment run:
```python
uv run run.py
```
To use different environments or algorithms, modify the `run.py` file. 
The package cremini_rl is based on the [mushroom_rl](https://github.com/MushroomRL/mushroom-rl) framework and contains the implementation of D-ATACOM as well as several Safe RL baselines. 
Currently `D-ATACOM`, `LagSAC`, `WCSAC`, `SafeLayerTD3`, `CBF-SAC`, `ATACOM`, `IQN-ATACOM`
## Adding new Environments
To run the algorithms on a new environment add it to the `build_mdp` function in `cremini_rl/experiment.py`. 
The environment should be a subclass of `mushroom_rl.core.Environment`. The environment `cremini_rl\envs\goal_navigation_env.py` is an example of a environment wrapper for safety gymnasium.

For `D-ATACOM`, `IQN-ATACOM`, `CBF-SAC` the dynamics of the agent are also required. They should be a subclass of `cremini_rl.dynamics.dynamics.ControlAffineSystem`.  


