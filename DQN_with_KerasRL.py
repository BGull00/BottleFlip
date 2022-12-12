"""
ECE 414 Final Project

Based heavily on example from Keras-RL
https://github.com/keras-rl/keras-rl/blob/master/examples/duel_dqn_cartpole.py

"""

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from mlagents_envs.environment import UnityEnvironment

from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

import tensorflow as tf
import datetime

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


channel = EngineConfigurationChannel()

# Load the environment using ML-Agents Python Low Level API
# unity_env = UnityEnvironment(side_channels=[channel])

channel.set_configuration_parameters(time_scale=20.0)

"""
# Reset the environment to start with a blank slate
unity_env.reset()

# Get environment details
behavior_names = unity_env.behavior_specs.keys()
"""
# Load the environment using ML-Agents Python Low Level API
unity_env = UnityEnvironment(side_channels=[channel])
env = UnityToGymWrapper(unity_env, flatten_branched=True, allow_multiple_obs=True)

"""

while True:
    env.step()
    for bname in behavior_names:
        decisionSteps, terminalSteps = env.get_steps(bname)
        for o in decisionSteps.obs:
            print(o)

        if len(decisionSteps) > 0:
            env.set_actions(bname, ActionTuple(np.empty([1, 0]), np.asarray([[1, 1, 0]])))

"""
#
# n_actions = 3
# state_dims = 4
#
# np.random.seed(123)
#
# # Build simple model
# model = Sequential()
# model.add(Flatten(input_shape=(1,) + (state_dims,)))
# model.add(Dense(16))
# model.add(Activation('relu'))
# model.add(Dense(16))
# model.add(Activation('relu'))
# model.add(Dense(16))
# model.add(Activation('relu'))
# model.add(Dense(n_actions, activation='linear'))
# print(model.summary())
#
# # Configure and compile an agent
# memory = SequentialMemory(limit=50000, window_length=1)
# policy = BoltzmannQPolicy()
#
# dqn = DQNAgent(model=model, nb_actions=n_actions, memory=memory, nb_steps_warmup=10,
#                target_model_update=1e-2, policy=policy)
# dqn.compile(Adam(lr=1e-3), metrics=['mae'])
#
# # Okay, now it's time to learn something! We visualize the training here for show, but this
# # slows down training quite a lot. You can always safely abort the training prematurely using
# # Ctrl + C.
# dqn.fit(env, nb_steps=50000, visualize=True, verbose=2)
#
# # After training is done, we save the final weights.
# dqn.save_weights('duel_dqn_{}_weights.h5f'.format("BottleFlipEnv"), overwrite=True)
#
# # Finally, evaluate our algorithm for 5 episodes.
# dqn.test(env, nb_episodes=5, visualize=True)
#
#

np.random.seed(123)

nb_actions = env.action_size

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + (1,4)))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))

model.build()
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.

# Set up for TensorBoard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = "logs/fit/" + current_time
test_log_dir = "logs/test/" + current_time
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=train_log_dir, histogram_freq=1)

dqn.fit(env, nb_steps=50000000000000, visualize=True, verbose=2)

# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights.h5f'.format("BottleFlipEnv"), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5000, visualize=True)
