# Reinforcement-Learning-using-OpenAI-Gym
Reinforcement Learning algorithms SARSA, Q-Learning, DQN, for Classical and MuJoCo Environments and testing them with OpenAI Gym.
### SARSA Cart Pole
SARSA (State-Action-Reward-State-Action) is a simple on-policy reinforcement learning
algorithm in which the agent tries to learn the optimal policy following the current policy (epsilon-greedy) generating action from current state and also the next state.

I have implemented SARSA for the Cart Pole problem, a classical environment provided by OpenAI gym. 

Problem Goal:<br/>
The Cart Pole Problem has 4 states at every time step, 
>[**the position of the cart on the horizontal
axis**, <br/>**the cart’s velocity on that same axis**, <br/>**the pole’s angular position on the cart**, <br/>**the angular
velocity of the pole on the cart**]

and there are 2 actions which the cart can take [**going to the left**,
**going to the right**]. 

The main goal is to balance the pole on the cart for the longest time taking
appropriate actions at every timestep.

Implementation<br/>
* I have discretize the 4 states into [2,2,8,4] discrete states respectively and have maintained a specific range of values for each of the states. 
* I have also used decaying exploration rate to decrease random exploration towards the end of the episodes.

### Q-Learning (SARSAMAX) Cart Pole
Q-Learning is a simple off-policy reinforcement learning
algorithm in which the agent tries to learn the optimal policy following the current policy (epsilon-greedy) generating action from current state and transitions to the state using the action which has the max Q-value, which is the why it is also called SARSAMAX.

I have implemented Q-learning for the Cart Pole problem, a classical environment provided by OpenAI gym. 

Implementation<br/>
* I have discretize the 4 states into [2,2,8,4] discrete states respectively and have maintained a specific range of values for each of the states. 
* I have also used decaying exploration rate to decrease random exploration towards the end of the episodes.
