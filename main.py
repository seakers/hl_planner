from spinup import ppo
import tensorflow as tf
import gym


def main():
    args = {
        "forest_data_path": "/Users/anmartin/Projects/summer_project/hl_planner/forest_data.tiff",
        "num_measurements": 6,
        "max_forest_heights": [60, 90, 45, 38, 30, 76],
        "orbit_altitude": 757000,
    }
    env_fn = lambda: gym.make('gym_orekit:online-orekit-v0', **args)

    ac_kwargs = dict(hidden_sizes=[64, 64], activation=tf.nn.relu)
    logger_kwargs = dict(output_dir='./output', exp_name='test1')
    ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=20160, epochs=10, max_ep_len=20160, save_freq=2, logger_kwargs=logger_kwargs)


if __name__ == '__main__':
    main()
