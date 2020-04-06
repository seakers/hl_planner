#from stable_baselines.common.policies import MlpPolicy
#from stable_baselines.common import make_vec_env
#from stable_baselines import PPO2
import os
import ray
from ray import tune

from gym_orekit.envs import OfflineOrekitEnv


def main():
    curdir = os.getcwd()
    args = {
        "forest_data_path": os.path.join(curdir, "forest_data.tiff"),
        "simulation_data_path": os.path.join(curdir, "fastsimulation.json"),
        "num_measurements": 6,
        "max_forest_heights": [60, 90, 45, 38, 30, 76],
        "orbit_altitude": 757000,
        "draw_plot": False
    }

    # multiprocess environment
    # env = make_vec_env(OfflineOrekitEnv, n_envs=8, env_kwargs=args)
    #
    # model = PPO2(MlpPolicy, env, verbose=1, n_steps=20162, nminibatches=8, cliprange=10000, tensorboard_log="./log_1/")
    # model.learn(total_timesteps=20162*3000)
    # model.save("plan_ppo")

    ray.init()
    tune.run(
        "PPO",
        stop={
            "training_iteration": 3000
        },
        config={
            "env": OfflineOrekitEnv,
            "num_workers": 4,
            #"lr": tune.grid_search([0.01, 0.001, 0.0001]),
            "batch_mode": "complete_episodes",
            "env_config": args,

            # Size of batches collected from each worker
            "sample_batch_size": 20162,
            # Number of timesteps collected for each SGD round
            "train_batch_size": 80648,
            "vf_clip_param": 5000,
            "num_gpus": 1
        },
        checkpoint_freq=100
    )

    # ac_kwargs = dict(hidden_sizes=[64, 64], activation=tf.nn.relu)
    # logger_kwargs = dict(output_dir='./output', exp_name='test1')
    # ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=20161, epochs=3000, max_ep_len=20161, save_freq=100, clip_ratio=0.5, pi_lr=1e-2, vf_lr=1e-2, logger_kwargs=logger_kwargs)


if __name__ == '__main__':
    main()
