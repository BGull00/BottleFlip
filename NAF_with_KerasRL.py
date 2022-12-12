"""
ECE 414 Final Project

"""

import numpy as np
import datetime
import rl.callbacks

from mlagents_envs.environment import UnityEnvironment

from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents import NAFAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.core import Processor


channel = EngineConfigurationChannel()
channel.set_configuration_parameters(time_scale=50.0)

# Load the environment using ML-Agents Python Low Level API
unity_env = UnityEnvironment(side_channels=[channel])
env = UnityToGymWrapper(unity_env, flatten_branched=True, allow_multiple_obs=True)

np.random.seed(123)
nb_actions = env.action_size

# Build all models
# Build V model
V_model = Sequential()
V_model.add(Flatten(input_shape=(1,) + (1,4)))
V_model.add(Dense(16))
V_model.add(Activation('relu'))
V_model.add(Dense(16))
V_model.add(Activation('relu'))
V_model.add(Dense(16))
V_model.add(Activation('relu'))
V_model.add(Dense(1))
V_model.add(Activation('linear'))
print(V_model.summary())

# Build mu model
mu_model = Sequential()
mu_model.add(Flatten(input_shape=(1,) + (1, 4)))
mu_model.add(Dense(16))
mu_model.add(Activation('relu'))
mu_model.add(Dense(16))
mu_model.add(Activation('relu'))
mu_model.add(Dense(16))
mu_model.add(Activation('relu'))
mu_model.add(Dense(nb_actions))
mu_model.add(Activation('linear'))
print(mu_model.summary())

# Build L model
action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + (1, 4), name='observation_input')
x = Concatenate()([action_input, Flatten()(observation_input)])
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(((nb_actions * nb_actions + nb_actions) // 2))(x)
x = Activation('linear')(x)
L_model = Model(inputs=[action_input, observation_input], outputs=x)
print(L_model.summary())

# Configure and compile our agent
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.3, size=nb_actions)
agent = NAFAgent(nb_actions=nb_actions, V_model=V_model, L_model=L_model, mu_model=mu_model,
                 memory=memory, nb_steps_warmup=100, random_process=random_process,
                 gamma=.99, target_model_update=1e-3, processor=processor)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
cb_filepath = "NAF_results/" + current_time
cb = rl.callbacks.FileLogger(filepath=cb_filepath)

# Train with DQN
agent.fit(env, nb_steps=50000, visualize=True, verbose=1, nb_max_episode_steps=200)

# Save the final weights
agent.fit(env, nb_steps=50000, visualize=True, verbose=1, nb_max_episode_steps=200)

# Evaluate
agent.fit(env, nb_steps=50000, visualize=True, verbose=1, nb_max_episode_steps=200)
