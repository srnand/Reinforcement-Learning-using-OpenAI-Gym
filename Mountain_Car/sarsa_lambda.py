import gym
import math
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

env = gym.make('MountainCar-v0')
Q_table = np.zeros((65,65,3))
alpha=0.3
buckets=[65, 65]
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

def updateQ_SARSA(observation,reward,action,ini_obs,next_action,t,eligibility):
	
	interval = toDiscreteStates(observation)

	Q_next = Q_table[tuple(interval)][next_action]

	ini_interval = toDiscreteStates(ini_obs)

	lr=max(0.4, min(0.1, 1.0 - math.log10((t+1)/125.)))

	td_error=(reward + gamma*(Q_next) - Q_table[tuple(ini_interval)][action])

	# print Q_table[tuple(ini_interval)][action], Q_table[tuple(ini_interval)][action]*eligibility[tuple(ini_interval)][action]

	# Q_table[tuple(ini_interval)][action]+=lr*td_error*eligibility[tuple(ini_interval)][action]

	# eligibility[tuple(ini_interval)][action]+=1

	Q_table[:,:,action]+=lr*td_error*(eligibility[:,:,action])
	# print Q_table[:,:,action]
	# print lr*td_error*eligibility[tuple(interval)][action]

# def updateQ_SARSA(observation,reward,action,ini_obs,next_action,t,eligibility):
# 	interval = toDiscreteStates(observation)
# 	Q_next = Q_table[tuple(interval)][next_action]
# 	ini_interval = toDiscreteStates(ini_obs)
	
# 	alpha=max(0.4, min(0.1, 1.0 - math.log10((t+1)/125.)))

# 	td= (reward + gamma*(Q_next) - Q_table[tuple(ini_interval)][action])

# 	Q_table[:,action]+=alpha*td#*(eligibility[:,action])

# rewards=[]
lambdaa=0.8

for i_episode in range(2500):
	observation = env.reset()
	t=0
	eligibility = np.zeros((65,65,3))
	while (True):
		# env.render()
		action = get_action(observation,i_episode)
		observation1, reward, done, info = env.step(action)

		interval = toDiscreteStates(observation)
		eligibility *= lambdaa * gamma
		# # print eligibility
		eligibility[tuple(interval)][action]+=1
		# print observation1

		next_action = get_action(observation1,i_episode)
		updateQ_SARSA(observation1,reward,action,observation,next_action,i_episode,eligibility)
		observation=observation1
		action = next_action
		t+=1
		if done:
			rewards.append(t+1)
			break

plt.plot(rewards)
plt.show()
