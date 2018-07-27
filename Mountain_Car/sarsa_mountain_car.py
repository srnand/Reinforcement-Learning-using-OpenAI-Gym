import gym
import math
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')
Q_table = np.zeros((20,20,3))
alpha=0.3
buckets=[20, 20]
gamma=0.99
rewards=[]

def toDiscreteStates(observation):
	interval=[0 for i in range(len(observation))]
	max_range=[1.2,0.07]	#[4.8,3.4*(10**38),0.42,3.4*(10**38)]

	for i in range(len(observation)):
		data = observation[i]
		inter = int(math.floor((data + max_range[i])/(2*max_range[i]/buckets[i])))
		if inter>=buckets[i]:
			interval[i]=buckets[i]-1
		elif inter<0:
			interval[i]=0
		else:
			interval[i]=inter
	return interval

def get_action(observation,t):

	if np.random.random()<max(0.001, min(0.015, 1.0 - math.log10((t+1)/220.))):#get_epsilon(t):
		return env.action_space.sample()
	interval = toDiscreteStates(observation)
	
	# if Q_table[tuple(interval)][0] >=Q_table[tuple(interval)][1]:
	# 	return 0
	# else:
	# 	return 1
	return np.argmax(np.array(Q_table[tuple(interval)]))

def updateQ_SARSA(observation,reward,action,ini_obs,next_action,t):
	
	interval = toDiscreteStates(observation)

	Q_next = Q_table[tuple(interval)][next_action]

	ini_interval = toDiscreteStates(ini_obs)

	Q_table[tuple(ini_interval)][action]+=max(0.4, min(0.1, 1.0 - math.log10((t+1)/125.)))*(reward + gamma*(Q_next) - Q_table[tuple(ini_interval)][action])


# print toDiscreteStates([1.4,2,3,0.4,2,4])

for i_episode in range(3000):
	observation = env.reset()
	t=0
	while (True):
		# env.render()
		# print(observation)
		#action = env.action_space.sample()
		action = get_action(observation,i_episode)
		observation1, reward, done, info = env.step(action)
		next_action = get_action(observation1,i_episode)
		# observation2, reward, done, info = env.step(next_action)
		# next2_action = get_action(observation2,i_episode)
		updateQ_SARSA(observation1,reward,action,observation,next_action,i_episode)
		# updateQ(observation1,reward,action,observation,t)
		observation=observation1
		action = next_action
		t+=1
		if done:
			# print("Episode finished after {} timesteps".format(t+1))
			rewards.append(t+1)
			break
# print rewards
# print Q_table
plt.plot(rewards)
plt.show()

