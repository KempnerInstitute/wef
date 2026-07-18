import time
import numpy as np
import torch

# from onpolicy.runner.shared.base_runner import Runner
import tqdm

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from ..base_runner import Runner # load module from parent folder
from base_runner_shared import Runner


from onpolicy.instrument.recorder import Recorder
from datetime import datetime
import string
import random

from utils_logging import (
    plot_train_metrics,
    plot_train_metrics_subrewards,
    plot_mean_cumulative_rewards,
)

from pathlib import Path
import imageio
import imageio.v3 as iio

def _t2n(x):
    return x.detach().cpu().numpy()


class MAFishRunner(Runner):
    """Runner class to perform training, evaluation, and data collection"""

    def __init__(self, config):
        super(MAFishRunner, self).__init__(config)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = (
            int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        )

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            reward_component_sums = {}
            reward_component_count = 0

            for step in range(self.episode_length):
                # Sample actions
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    actions_env,
                ) = self.collect(step)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)

                for env_info in infos:
                    for agent_id in range(self.num_agents):
                        if isinstance(env_info, dict):
                            agent_info = env_info.get(agent_id, {})
                        elif isinstance(env_info, (list, tuple)):
                            agent_info = env_info[agent_id] if agent_id < len(env_info) else {}
                        else:
                            agent_info = {}
                        components = agent_info.get("reward_components")
                        if components is None:
                            continue
                        reward_component_count += 1
                        for k, v in components.items():
                            reward_component_sums[k] = (
                                reward_component_sums.get(k, 0.0) + float(v)
                            )

                data = (
                    obs,
                    rewards,
                    dones,
                    infos,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )

                # Insert data into buffer
                self.insert(data)

            # Compute return and update network
            self.compute()
            train_infos = self.train()

            # Post process
            total_num_steps = (
                (episode + 1) * self.episode_length * self.n_rollout_threads
            )

            # Save the initial model
            if episode == 0:
                self.save(episode="initial")  # Abusing what is normally an int
            # Save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save(episode=episode)

            # Log information
            if episode % self.log_interval == 0:
                end = time.time()
                print(
                    "\n Env [{}], Algo {}, Exp {}, updates {}/{} ep, timesteps {}/{}, FPS {}".format(
                        self.env_name,
                        self.algorithm_name,
                        self.experiment_name,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                    )
                )

                env_infos = {}
                for agent_id in range(self.num_agents):
                    idv_rews = []
                    for info in infos:
                        if "individual_reward" in info[agent_id].keys():
                            idv_rews.append(info[agent_id]["individual_reward"])
                    agent_k = "agent%i/individual_rewards" % agent_id
                    env_infos[agent_k] = idv_rews

                train_infos["average_episode_rewards"] = (
                    np.mean(self.buffer.rewards) * self.episode_length
                )
                if reward_component_count > 0:
                    train_infos["r_components"] = {
                        f"r_{k}": v / reward_component_count
                        for k, v in reward_component_sums.items()
                    }
                print(
                    "average episode rewards is {}".format(
                        train_infos["average_episode_rewards"]
                    )
                )
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # Eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

        plot_train_metrics(self.log_dir)
        plot_train_metrics_subrewards(self.log_dir)
        try:
            plot_mean_cumulative_rewards(
                os.path.join(self.log_dir, "env_params.csv"),
                smoothing_window=20
            )
        except Exception as e:
            print("Error plotting mean cumulative rewards:", e)


    def warmup(self):
        # Reset env
        obs = self.envs.reset()

        # Replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic = (
            self.trainer.policy.get_actions(
                np.concatenate(self.buffer.share_obs[step]),
                np.concatenate(self.buffer.obs[step]),
                np.concatenate(self.buffer.rnn_states[step]),
                np.concatenate(self.buffer.rnn_states_critic[step]),
                np.concatenate(self.buffer.masks[step]),
            )
        )
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(
            np.split(_t2n(action_log_prob), self.n_rollout_threads)
        )
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_states_critic), self.n_rollout_threads)
        )
        # Rearrange action
        actions_env = actions

        return (
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            actions_env,
        )

    def insert(self, data):
        (
            obs,
            rewards,
            dones,
            infos,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(
            share_obs,
            obs,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            rewards,
            masks,
        )

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]),
            dtype=np.float32,
        )
        eval_masks = np.ones(
            (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
        )

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=True,
            )
            eval_actions = np.array(
                np.split(_t2n(eval_action), self.n_eval_rollout_threads)
            )
            eval_rnn_states = np.array(
                np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads)
            )

            eval_actions_env = eval_actions

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(
                eval_actions_env
            )
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )
            eval_masks = np.ones(
                (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
            )
            eval_masks[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), 1), dtype=np.float32
            )

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos["eval_average_episode_rewards"] = np.sum(
            np.array(eval_episode_rewards), axis=0
        )
        eval_average_episode_rewards = np.mean(
            eval_env_infos["eval_average_episode_rewards"]
        )
        print(
            "eval average episode rewards of agent: "
            + str(eval_average_episode_rewards)
        )
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        if hasattr(self.all_args, "run_name"):
            run_name = self.all_args.run_name
        else:
            run_name = "".join(
                random.choices(string.ascii_letters + string.digits, k=8)
            )
            # because args are saved at runner initialization,
            # this is not actually getting saved to all_args.json
            # but is available in the self.all_args object
            self.all_args.run_name = run_name

        """Visualize the env."""
        envs = self.envs
        seeds = [envs.envs[i].env_seed for i in range(self.n_rollout_threads)]
        num_render_envs = max(1, min(getattr(self.all_args, 'num_render_envs', 1), self.n_rollout_threads))

        # Keys that belong in arena / episode records, not per-agent behavior
        _ARENA_KEYS = frozenset({"food_positions", "active_agent_ids", "muted_agent_ids"})
        _EPISODE_KEYS = frozenset({
            "arena_type", "patch_kwargs", "arena_size", "active_agent_ids", "muted_agent_ids",
            "rw_eod_A", "rw_freeze_A", "rw_size_A", "rw_size_B",
            "rw_B_x", "rw_B_y", "rw_B_orientation", "rw_trial_id",
        })
        _SKIP_BEHAVIOR = _ARENA_KEYS | _EPISODE_KEYS

        evals_root = os.path.join(str(self.run_dir), "evals")
        os.makedirs(evals_root, exist_ok=True)
        summary_csv = os.path.join(evals_root, "eval_summary.csv")

        video_writers = [None] * num_render_envs
        for episode in range(self.all_args.render_episodes):  # render_episodes = num_eval_rollouts; 0 means no pkls written
            recorder = Recorder()

            episode_seeds = [
                seed + 197 * episode + 1 for seed in seeds
            ]  # 197x and 1 are arbitrary offsets
            obs = envs.reset_with_seeds(episode_seeds)

            if self.all_args.save_vids and episode < self.all_args.num_vids_to_save:
                if hasattr(self.trainer.policy.actor.base, "last_attn_mask"):
                    attentions = np.zeros(
                        (
                            self.n_rollout_threads,
                            self.num_agents,
                            self.envs.observation_space[0].shape[0],
                        )
                    )
                    # handle single-threaded case
                    if attentions.ndim == 3 and attentions.shape[0] == 1:
                        attentions = attentions[0]
                    auxs = ["eods", "spis", "observations", "attentions"]
                    frames = envs.render("rgb_array", auxs=auxs, attentions=attentions, num_envs=num_render_envs)
                else:
                    frames = envs.render("rgb_array", num_envs=num_render_envs)

                os.makedirs(self.output_dir, exist_ok=True)
                for env_i, frame in enumerate(frames):
                    if video_writers[env_i] is None:
                        video_writer_path = f"{self.output_dir}/ep{episode}_env{env_i}.mp4"
                        video_writers[env_i] = imageio.get_writer(
                            video_writer_path,
                            format="ffmpeg",
                            mode="I",
                            fps=int(1.0 / self.all_args.ifi),
                        )
                        print(f"Opened video writer at {video_writer_path}")
                    video_writers[env_i].append_data(frame)

            if self.all_args.save_svg_snapshots:
                os.makedirs(f"{self.output_dir}/svg_snapshots", exist_ok=True)
                envs.save_svg_snapshot(f"{self.output_dir}/svg_snapshots/ep{episode}_step0.svg")

            rnn_states = np.zeros(
                (
                    self.n_rollout_threads,
                    self.num_agents,
                    self.recurrent_N,
                    self.hidden_size,
                ),
                dtype=np.float32,
            )
            masks = np.ones(
                (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
            )

            episode_rewards = []

            for step in tqdm.tqdm(range(self.episode_length)):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(
                    np.concatenate(obs),
                    np.concatenate(rnn_states),
                    np.concatenate(masks),
                    deterministic=True,
                )
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(
                    np.split(_t2n(rnn_states), self.n_rollout_threads)
                )

                actions_env = actions

                # Observe reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                if hasattr(self.trainer.policy.actor.base, "last_attn_mask"):
                    attn_mask = (
                        self.trainer.policy.actor.base.last_attn_mask
                    )  # shape [batch_size * num_agents, obs_dim]
                    try:
                        attn_mask = attn_mask.reshape(
                            self.n_rollout_threads, self.num_agents, -1
                        ).numpy()
                    except Exception as e:
                        print(
                            f"[attn_mask reshape error] Expected shape: ({self.n_rollout_threads * self.num_agents}, -1), got: {attn_mask.shape}"
                        )
                        attn_mask = None
                else:
                    attn_mask = None

                # Record structured data via Recorder
                for env_i, env_infos in enumerate(infos):
                    a0_info = env_infos[0]

                    # Episode metadata — logged once per (env_id, episode) at step 0
                    if step == 0:
                        episode_meta = {k: a0_info[k] for k in _EPISODE_KEYS if k in a0_info}
                        recorder.record_episode(
                            env_id=env_i, ep=episode,
                            seed=episode_seeds[env_i],
                            eval_run_name=run_name,
                            **episode_meta,
                        )

                    # Arena state — one row per (env_id, ep, t)
                    arena_fields = {k: a0_info[k] for k in _ARENA_KEYS if k in a0_info}
                    recorder.record_arena(env_id=env_i, ep=episode, t=step, **arena_fields)

                    # Per-agent behavior — one row per (env_id, ep, t, agent_id)
                    for agent_j, agent_info in enumerate(env_infos):
                        behavior_fields = {k: v for k, v in agent_info.items()
                                           if k not in _SKIP_BEHAVIOR}
                        recorder.record_behavior(
                            env_id=env_i, ep=episode, t=step, agent_id=agent_j,
                            actions=actions_env[env_i, agent_j],
                            rewards=float(rewards[env_i, agent_j, 0]),
                            **behavior_fields,
                        )

                # Dense arrays stored separately — gated by EvalSpec flags
                if getattr(self.all_args, 'save_obs', False):
                    recorder.record_obs(obs)
                if getattr(self.all_args, 'save_rnn', False):
                    recorder.record_rnn(rnn_states)
                if getattr(self.all_args, 'save_attn', False) and attn_mask is not None:
                    recorder.record_attn(attn_mask)

                rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                masks = np.ones(
                    (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
                )
                masks[dones == True] = np.zeros(
                    ((dones == True).sum(), 1), dtype=np.float32
                )

                if self.all_args.save_vids and episode < self.all_args.num_vids_to_save:
                    if attn_mask is not None:
                        # handle single-threaded case differently
                        if attn_mask.ndim == 3 and attn_mask.shape[0] == 1:
                            attn_mask = attn_mask[0]
                        auxs = ["eods", "spis", "observations", "attentions"]
                        frames = envs.render(
                            "rgb_array", auxs=auxs, attentions=attn_mask, num_envs=num_render_envs
                        )
                    else:
                        frames = envs.render("rgb_array", num_envs=num_render_envs)
                    for env_i, frame in enumerate(frames):
                        video_writers[env_i].append_data(frame)

                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    # if elapsed < self.all_args.ifi:
                    #     time.sleep(self.all_args.ifi - elapsed)

            # Save the per-episode bundle (behavior, arena, episodes, rnn, attn)
            # and append a row to eval_summary.csv
            stem = f"{self.output_dir}/ep{episode}"
            recorder.save(stem, summary_csv=summary_csv)
            print(f"Saved bundle: {stem}_{{behavior,arena,episodes}}.{{pkl,json}}")

            if self.all_args.save_vids and episode < self.all_args.num_vids_to_save:
                for env_i, writer in enumerate(video_writers):
                    if writer is not None:
                        writer.close()
                video_writers = [None] * num_render_envs

            print(
                "average episode rewards is: "
                + str(np.mean(np.sum(np.array(episode_rewards), axis=0)))
            )

        return run_name
