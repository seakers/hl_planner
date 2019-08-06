from gym_orekit.envs import OfflineOrekitEnv
from ray.tune import register_env
from spinup.utils.test_policy import load_policy, run_policy
import ray.rllib.rollout as rollout


def main():
    env_args = {
        "forest_data_path": "/Users/anmartin/Projects/summer_project/hl_planner/forest_data.tiff",
        "simulation_data_path": "/Users/anmartin/Projects/FormationSimulation/fastsimulation.json",
        "num_measurements": 6,
        "max_forest_heights": [60, 90, 45, 38, 30, 76],
        "orbit_altitude": 757000,
        "draw_plot": True
    }

    parser = rollout.create_parser()
    args = parser.parse_args()

    register_env("offline-orekit", lambda _: OfflineOrekitEnv(env_args))

    rollout.run(args, parser)


if __name__ == '__main__':
    main()
