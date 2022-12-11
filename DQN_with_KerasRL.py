<<<<<<< HEAD

"""
ECE 414 Final Project

Based heavily on example from Keras-RL
https://github.com/keras-rl/keras-rl/blob/master/examples/duel_dqn_cartpole.py
"""

import numpy as np

from mlagents_envs.environment import UnityEnvironment

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from mlagents_envs.environment import UnityEnvironment

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

ENV_NAME = "BottleFlipEnv"

# Load the environment using ML-Agents Python Low Level API
#
env = UnityEnvironment(file_name="BottleFlipEnv")

# Reset the environment to start with a blank slate
env.reset()

# Get environment details
behavior_name = env.get_behavior_names()[0]
behavior_spec = env.get_behavior_spec(behavior_name)

n_actions = behavior_spec.action_size
state_dims = np.sum(behavior_spec.observation_shapes)  # list[tuple(52,), tuple(2,)] => 54 total obs

if behavior_spec.is_action_continuous():
    print("Action space is CONTINUOUS")
else:
    print("Action space is DISCRETE")
    print(behavior_spec.discrete_action_branches)


np.random.seed(123)
env.seed(123)

# Build simple model
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(n_actions, activation='linear'))
print(model.summary())

# Configure and compile an agent
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
# enable the dueling network
# you can specify the dueling_type to one of {'avg','max','naive'}
dqn = DQNAgent(model=model, nb_actions=n_actions, memory=memory, nb_steps_warmup=10,
               enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=50000, visualize=True, verbose=2)

# After training is done, we save the final weights.
dqn.save_weights('duel_dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=True)
=======

"""
ECE 414 Final Project

Based heavily on example from Keras-RL
https://github.com/keras-rl/keras-rl/blob/master/examples/duel_dqn_cartpole.py
"""

import numpy as np

from mlagents_envs.environment import UnityEnvironment

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from mlagents_envs.environment import UnityEnvironment

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

ENV_NAME = "BottleFlipEnv"

# Load the environment using ML-Agents Python Low Level API
#
env = UnityEnvironment(file_name="BottleFlipEnv")

# Reset the environment to start with a blank slate
env.reset()

# Get environment details
behavior_name = env.get_behavior_names()[0]
behavior_spec = env.get_behavior_spec(behavior_name)

n_actions = behavior_spec.action_size
state_dims = np.sum(behavior_spec.observation_shapes)  # list[tuple(52,), tuple(2,)] => 54 total obs

if behavior_spec.is_action_continuous():
    print("Action space is CONTINUOUS")
else:
    print("Action space is DISCRETE")
    print(behavior_spec.discrete_action_branches)


np.random.seed(123)
env.seed(123)

# Build simple model
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(n_actions, activation='linear'))
print(model.summary())

# Configure and compile an agent
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
# enable the dueling network
# you can specify the dueling_type to one of {'avg','max','naive'}
dqn = DQNAgent(model=model, nb_actions=n_actions, memory=memory, nb_steps_warmup=10,
               enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=50000, visualize=True, verbose=2)

# After training is done, we save the final weights.
dqn.save_weights('duel_dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=True)
>>>>>>> e970ba12a2b44054a6155842085cc318c27a79f9
