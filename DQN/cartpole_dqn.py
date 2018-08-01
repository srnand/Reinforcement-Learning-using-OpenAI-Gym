import gym
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, Dense
from collections import deque
import random
from random import randint
import numpy as np
import skimage.color, skimage.transform,  skimage.exposure
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import seaborn as sns

import time
class DQAgent:
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.input_size=self.env.observation_space.shape[0]
        self.action_size=self.env.action_space.n
        self.nets=[]
        for i in range(self.action_size):
            self.nets.append(self.qnetwork(self.input_size))
            
        self.memory=deque(maxlen=2000)
        self.batch_size=32
        self.epsilon = 1.0
        self.gamma=0.95
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
    def qnetwork(self,input_size):
        model = Sequential()
        model.add(Dense(24, input_dim=input_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse',optimizer=Adam(lr=0.001))
        return model
    
    def remember(self,s,a,r,s_,d):
        self.memory.append((s,a,r,s_,d))
    
    def get_action(self,state):
        if self.epsilon>np.random.rand():
            return random.randrange(self.action_size)
        val=[]
        for i in self.nets:
#             print i.predict(state)
            val.append(i.predict(state)[0])
#         actions = self.dqn.predict(state)
#         print val
        return np.argmax(val)
    
    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        for s,a,r,s_,d in minibatch:
            if d:
                target=r
            else:
                val=[]
                for i in self.nets:
                    val.append(i.predict(s_)[0])
                target = r+self.gamma*np.amax(val)
            self.nets[a].fit(s,[[target]],epochs=1,verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
if __name__ == "__main__":
    start=time.time()
    agent = DQAgent()
    timesteps=[]
    for e in range(1000):
        t=0
        state=agent.env.reset()
        state=np.reshape(state,[1,agent.input_size])
        print e
        while (True):
#             agent.env.render()
            action = agent.get_action(state)
            next_state, reward, done, _ = agent.env.step(action)
            if done:
                if t!=200:
                    reward=-10
            next_state=np.reshape(next_state,[1,agent.input_size])
            agent.remember(state,action,reward,next_state,done)
            state=next_state
            t+=1
#             print done
            if done:
                timesteps.append(t)
                break
            if len(agent.memory)>agent.batch_size:
                agent.replay()
    end=time.time()
    print ("--- %s seconds ---" % (end - start))
    plt.plot(timesteps)
    plt.show()