"""
ECE 414 Final Project

Based heavily on example from Keras-RL
https://github.com/keras-rl/keras-rl/blob/master/examples/duel_dqn_cartpole.py

"""

import numpy as np
import rl.callbacks

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
channel.set_configuration_parameters(time_scale=50.0)

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

np.random.seed(123)
nb_actions = env.action_size

# Build a simple model
model = Sequential()
model.add(Flatten(input_shape=(1,) + (1, 4)))
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

# Configure and compile our agent
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Set up for TensorBoard - not currently working
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = "logs/fit/" + current_time
test_log_dir = "logs/test/" + current_time
train_tb = tf.keras.callbacks.TensorBoard(log_dir=train_log_dir, histogram_freq=1)
test_tb = tf.keras.callbacks.TensorBoard(log_dir=test_log_dir, histogram_freq=1)

cb_filepath = "DQN_results/" + current_time
cb = rl.callbacks.FileLogger(filepath=cb_filepath)

# Train with DQN
dqn.fit(env, nb_steps=15000000, visualize=True, verbose=2, callbacks=[cb])

# Save the final weights
dqn.save_weights('dqn_{}_weights.h5f'.format("BottleFlipEnv"), overwrite=True)

# Evaluate
dqn.test(env, nb_episodes=5000, visualize=True)
