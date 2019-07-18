from spinup import ppo
import tensorflow as tf
import gym

env_fn = lambda : gym.make('gym_orekit:orekit-v0')

ac_kwargs = dict(hidden_sizes=[64, 64], activation=tf.nn.relu)

logger_kwargs = dict(output_dir='./output', exp_name='test1')

ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=20160, epochs=10, max_ep_len=20160, save_freq=2, logger_kwargs=logger_kwargs)