import ray
from ray import tune

from gym_orekit.envs import OfflineOrekitEnv


def main():
    args = {
        "forest_data_path": "/Users/anmartin/Projects/summer_project/hl_planner/forest_data.tiff",
        "simulation_data_path": "/Users/anmartin/Projects/FormationSimulation/fastsimulation.json",
        "num_measurements": 6,
        "max_forest_heights": [60, 90, 45, 38, 30, 76],
        "orbit_altitude": 757000,
        "draw_plot": False
    }

    ray.init()
    tune.run(
        "PPO",
        stop={
            "training_iteration": 3000
        },
        config={
            "env": OfflineOrekitEnv,
            "num_workers": 3,
            #"lr": tune.grid_search([0.01, 0.001, 0.0001]),
            "batch_mode": "complete_episodes",
            "env_config": args,

            # Size of batches collected from each worker
            "sample_batch_size": 20162,
            # Number of timesteps collected for each SGD round
            "train_batch_size": 60486,
            "vf_clip_param": 5000
        },
        checkpoint_freq=100
    )

    # ac_kwargs = dict(hidden_sizes=[64, 64], activation=tf.nn.relu)
    # logger_kwargs = dict(output_dir='./output', exp_name='test1')
    # ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=20161, epochs=3000, max_ep_len=20161, save_freq=100, clip_ratio=0.5, pi_lr=1e-2, vf_lr=1e-2, logger_kwargs=logger_kwargs)


if __name__ == '__main__':
    main()
