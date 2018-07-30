# Reinforcement-Learning-using-OpenAI-Gym
Reinforcement Learning algorithms SARSA, Q-Learning, DQN, for Classical and MuJoCo Environments and testing them with OpenAI Gym.
### SARSA Cart Pole
SARSA (State-Action-Reward-State-Action) is a simple on-policy reinforcement learning
algorithm in which the agent tries to learn the optimal policy following the current policy (epsilon-greedy) generating action from current state and also the next state.

Implemented SARSA for the Cart Pole problem, a classical environment provided by OpenAI gym. 

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
* Discretized the 4 states into [2,2,8,4] discrete states respectively and have maintained a specific range of values for each of the states. 
* Used decaying exploration rate to decrease random exploration towards the end of the episodes.

### Q-Learning (SARSAMAX) Cart Pole
Q-Learning is a simple off-policy reinforcement learning
algorithm in which the agent tries to learn the optimal policy following the current policy (epsilon-greedy) generating action from current state and transitions to the state using the action which has the max Q-value, which is the why it is also called SARSAMAX.

Implemented Q-learning for the Cart Pole problem, a classical environment provided by OpenAI gym. 

Implementation<br/>
* Discretized the 4 states into [2,2,8,4] discrete states respectively and have maintained a specific range of values for each of the states. 
* Used decaying exploration rate to decrease random exploration towards the end of the episodes.

Results:
<br/>
It looks from the graph that, the cart is able to balance the pole for the required amount of time almost constantly in about less than 2000 episodes for both algorithms. 

### SARSA Mountain Car

Problem Goal:<br/>
The Mountain Car Problem has 2 states at every time step, 
>[**the position of the car**, <br/>**the car’s velocity**]

and there are 3 actions which the cart can take [**going to the left**, **no action**,
**going to the right**]. 

The main goal is to make the car reach the goal(up-hill) taking
appropriate actions at every timestep.

Implementation<br/>
* Discretized the 2 states into [20,20] discrete states respectively and have maintained a specific range of values for each of the states. 
* Used decaying exploration rate to decrease random exploration towards the end of the episodes.
* Used gradually increasing learning rate because as the exploration rate decreases, confidence level increases and more learning happens towards the end of the episodes.

### Q-Learning (SARSAMAX) Mountain Car

Implementation<br/>
* Discretized the 2 states into [20,20] discrete states respectively and have maintained a specific range of values for each of the states. 
* Used decaying exploration rate to decrease random exploration towards the end of the episodes.
* Used gradually increasing learning rate because as the exploration rate decreases, confidence level increases and more learning happens towards the end of the episodes.

Results:
<br/>
It looks from the graph that, the car is able to reach the goal almost constantly in about less than 3000 episodes for both algorithms. 

### SARSA Mountain Car with Backward View (Eligibility Traces)

Implementation<br/>
* Discretized the 2 states into [65,65] discrete states respectively and have maintained a specific range of values for each of the states. 
* Used Eligiblity Traces, tuning value for lambda.
* Used decaying exploration rate to decrease random exploration towards the end of the episodes.
* Used gradually increasing learning rate because as the exploration rate decreases, confidence level increases and more learning happens towards the end of the episodes.

