# lunarlander
A Repo containing various RL algorithms and work on the Lunar Lander Domain

## About
Lunar Lander is a classical RL domain to test RL algorithms. I use [Gymnasium](https://gymnasium.farama.org/ "Gymnasium's Homepage") to simulate the environment. More info about the specifics of the environment can be found over here [Lunar Lander](https://gymnasium.farama.org/environments/box2d/lunar_lander/ "Lunar Lander"). 


## Setup
Install Gymnasium, PyTorch and PyGame

```pip install -r requirements.txt ```

## Try out the game yourself first!
One can try out playing the game by themselves by running

```python human_tryouts.py ```

To play, press Q for action 1 (Left Orientation Engine), W for action 2 (Main Engine), E for action 3 (Right Orientation Engine). Be sure to press only one key at a time as the environment only supports one action at a time. 

Try to get a score above 0! 

A landing (Safe or Crash) in between the flags gives a reward of +100. 

## Heuristic Landing

While going through the source code of Lunar Lander, I came across a [Heuristic Function](https://github.com/openai/gym/blob/dcd185843a62953e27c2d54dc8c2d647d604b635/gym/envs/box2d/lunar_lander.py#L726 "Heuristic Function") which worked based on applying engine forces and actions based on some predetermined angles, positions, and velocities. It performs quite well and gave an average total reward of 243. This motivated me to build an RL agent that could beat this score and also act as a goal that the RL agent should reach.

# RL Algos
(To Update Readme)

## SARSA

### SARSA Semi Gradient

### SARSA N-STEP Semi Gradient

## Policy Gradient Methods

# Further Ideas:
Adding Experience Replay Buffer and testing various buffer sizes and types like prioritized experience.

Policy Gradient Methods

REINFORCE and Actor - Crtitic Methods 

Off-policy initialization from human play and training on that

RAINBOW

