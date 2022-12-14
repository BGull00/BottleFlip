
from stable_baselines3 import A2C
import datetime

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper


channel = EngineConfigurationChannel()
channel.set_configuration_parameters(time_scale=50.0)

# Load the environment using ML-Agents Python Low Level API
unity_env = UnityEnvironment(side_channels=[channel])
env = UnityToGymWrapper(unity_env)

# Set up for TensorBoard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = "a2c_results/fit-" + current_time

# Do the training
model = A2C("MlpPolicy", env, learning_rate=0.00003, use_rms_prop=False, verbose=1, tensorboard_log=train_log_dir)
model.learn(total_timesteps=15000000)
model.save("A2C_BottleFlipEnv")

