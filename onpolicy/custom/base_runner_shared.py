import os
import json
import csv
import numpy as np
import torch
from tensorboardX import SummaryWriter
from onpolicy.utils.shared_buffer import SharedReplayBuffer


def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()


class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """

    def __init__(self, config):

        self.all_args = config["all_args"]
        self.envs = config["envs"]
        self.eval_envs = config["eval_envs"]
        self.device = config["device"]
        self.num_agents = config["num_agents"]
        if config.__contains__("render_envs"):
            self.render_envs = config["render_envs"]

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir

        self.run_dir = config["run_dir"]
        self.log_dir = self._create_subdir("logs")
        self.writter = SummaryWriter(self.log_dir) if self.all_args.log_tb else None
        self.save_dir = self._create_subdir("models")
        self.output_dir = self._create_subdir(config.get("output_subdir", "outputs"))
        self.write_args_to_file = (
            config["write_args_to_file"] if "write_args_to_file" in config else True
        )
        if self.write_args_to_file:
            self._save_all_args()
            self._save_env_args()
            # CSV log writing
            self._init_csv_log()

        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            from onpolicy.algorithms.mat.mat_trainer import MATTrainer as TrainAlgo
            from onpolicy.algorithms.mat.algorithm.transformer_policy import (
                TransformerPolicy as Policy,
            )
        else:
            from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
            from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import (
                R_MAPPOPolicy as Policy,
            )

        share_observation_space = (
            self.envs.share_observation_space[0]
            if self.use_centralized_V
            else self.envs.observation_space[0]
        )

        # print("obs_space: ", self.envs.observation_space)
        # print("share_obs_space: ", self.envs.share_observation_space)
        # print("act_space: ", self.envs.action_space)

        # policy network
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.policy = Policy(
                self.all_args,
                self.envs.observation_space[0],
                share_observation_space,
                self.envs.action_space[0],
                self.num_agents,
                device=self.device,
            )
        else:
            self.policy = Policy(
                self.all_args,
                self.envs.observation_space[0],
                share_observation_space,
                self.envs.action_space[0],
                device=self.device,
            )

        actor_path = os.path.join(self.save_dir, "actor.pt")
        if os.path.exists(actor_path):
            print(f"Found existing actor checkpoint at {actor_path}; loading model...\n")
            self.restore(self.save_dir)
        else:
            print(f"No saved actor at {actor_path}. Skipping restore.\n")

        # algorithm
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.trainer = TrainAlgo(
                self.all_args, self.policy, self.num_agents, device=self.device
            )
        else:
            self.trainer = TrainAlgo(self.all_args, self.policy, device=self.device)

        # buffer
        self.buffer = SharedReplayBuffer(
            self.all_args,
            self.num_agents,
            self.envs.observation_space[0],
            share_observation_space,
            self.envs.action_space[0],
        )

    def _init_csv_log(self):
        csv_path = os.path.join(self.log_dir, "train_metrics.csv")
        self.csv_file = open(csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        # write header row
        self.csv_writer.writerow(
            [
                "step",
                "average_episode_reward",
                "policy_loss",
                "value_loss",
                "dist_entropy",
                "actor_grad_norm",
                "critic_grad_norm",
                "ratio",
                "r_components_json",
            ]
        )

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            next_values = self.trainer.policy.get_values(
                np.concatenate(self.buffer.share_obs[-1]),
                np.concatenate(self.buffer.obs[-1]),
                np.concatenate(self.buffer.rnn_states_critic[-1]),
                np.concatenate(self.buffer.masks[-1]),
            )
        else:
            next_values = self.trainer.policy.get_values(
                np.concatenate(self.buffer.share_obs[-1]),
                np.concatenate(self.buffer.rnn_states_critic[-1]),
                np.concatenate(self.buffer.masks[-1]),
            )
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    def train(self):
        """Train policies with data in buffer."""
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)
        self.buffer.after_update()
        return train_infos

    def save(self, episode=0):
        """Save policy's actor and critic networks."""
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.policy.save(self.save_dir, episode)
        else:
            policy_actor = self.trainer.policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
            torch.save(
                policy_actor.state_dict(), str(self.save_dir) + f"/actor_{episode}.pt"
            )

            policy_critic = self.trainer.policy.critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")
            torch.save(
                policy_critic.state_dict(), str(self.save_dir) + f"/critic_{episode}.pt"
            )
            print(
                f"Saved actor and critic networks to {self.save_dir} at episode {episode}"
            )

    def restore(self, model_dir):
        """Restore policy's networks from a saved model."""
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.policy.restore(model_dir)
        else:
            policy_actor_state_dict = torch.load(str(self.save_dir) + "/actor.pt")
            self.policy.actor.load_state_dict(policy_actor_state_dict)
            if not self.all_args.use_render:
                policy_critic_state_dict = torch.load(
                    str(self.save_dir) + "/critic.pt"
                )
                self.policy.critic.load_state_dict(policy_critic_state_dict)

    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        if self.writter is not None:
            for k, v in train_infos.items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        tag = f"{k}/{sub_k}"
                        self.writter.add_scalars(tag, {tag: sub_v}, total_num_steps)
                    continue
                self.writter.add_scalars(k, {k: v}, total_num_steps)

        self.log_train_csv(train_infos, total_num_steps)

    def log_train_csv(self, train_infos, total_num_steps):
        avg_r = train_infos.get("average_episode_rewards")
        p_loss = train_infos.get("policy_loss")
        v_loss = train_infos.get("value_loss")
        dist_entropy = train_infos.get("dist_entropy")
        actor_grad_norm = train_infos.get("actor_grad_norm").item()
        critic_grad_norm = train_infos.get("critic_grad_norm").item()
        ratio = train_infos.get("ratio").item()
        r_components = train_infos.get("r_components", {})
        r_components_json = json.dumps(r_components, sort_keys=True)
        self.csv_writer.writerow(
            [
                total_num_steps,
                avg_r,
                p_loss,
                v_loss,
                dist_entropy,
                actor_grad_norm,
                critic_grad_norm,
                ratio,
                r_components_json,
            ]
        )
        self.csv_file.flush()

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        if self.writter is not None:
            for k, v in env_infos.items():
                if len(v) > 0:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)

    def _create_subdir(self, sub_dir_name):
        dir_path = str(self.run_dir / sub_dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        return dir_path

    def _save_all_args(self):
        all_args_filename = f"{self.log_dir}/all_args.json"
        all_args_dict = vars(self.all_args)

        with open(all_args_filename, "w") as f:
            json.dump(all_args_dict, f, indent=4)

        print(f"Saved all_args to {self.log_dir}")

    def _get_env_args(self):
        if not hasattr(self.envs, "envs"):
            print("Skipping saving env args due to multi-threaded eval")
            return
        multi_agent_env = self.envs.envs[0]
        agent_env_args = multi_agent_env.agent_objects[0].__dict__

        arena_env_args = multi_agent_env.arena.__dict__
        agent_env_args = self._get_json_serializable_version(agent_env_args)
        multi_agent_env_args = self._get_json_serializable_version(
            multi_agent_env.__dict__
        )
        try:
            agent_electrics_args = multi_agent_env.sensing_model.agent_electrics.__dict__
            agent_electrics_args = self._get_json_serializable_version(agent_electrics_args)
            multi_agent_env_args["sensing_model"] = {}
            multi_agent_env_args["sensing_model"]["agent_electrics"] = agent_electrics_args
        except Exception as e:
            print(f"Error serializing agent_electrics: {e}")

        arena_env_args = self._get_json_serializable_version(arena_env_args)

        return agent_env_args, multi_agent_env_args, arena_env_args

    def _save_env_args(self):
        agent_env_args, multi_agent_env_args, arena_env_args = self._get_env_args()

        env_args_dict = {
            "agent_env_args": agent_env_args,
            "multi_agent_env_args": multi_agent_env_args,
            "arena_env_args": arena_env_args,
        }

        env_args_filename = f"{self.log_dir}/env_args.json"
        with open(env_args_filename, "w") as f:
            json.dump(env_args_dict, f, indent=4)

        print(f"Saved env_args to {self.log_dir}")

    def _get_json_serializable_version(self, x):
        if isinstance(x, dict):
            return {k: self._get_json_serializable_version(v) for k, v in x.items()}
        elif isinstance(x, (list, tuple, set, frozenset)):
            return [self._get_json_serializable_version(v) for v in x]
        elif isinstance(x, np.ndarray):
            return x.tolist()
        elif self._is_json_serializable(x):
            return x
        else:
            return "unserializable"

    def _is_json_serializable(self, value):
        serializable_types = (type(None), bool, int, float, str, list, dict)
        if isinstance(value, serializable_types):
            return True
        try:
            json.dumps(value)
            return True
        except (TypeError, OverflowError):
            return False
