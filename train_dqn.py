import argparse
from collections import deque
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from dqn_model import DQNModel
from env import make_env
from rl_model import RLModel


class ReplayBuffer:
    def __init__(self, capacity, obs_dim):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_obs[self.pos] = next_obs
        self.dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            idx,
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
            np.ones(batch_size, dtype=np.float32),
        )

    def update_priorities(self, idx, priorities):
        pass


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, obs_dim, alpha=0.6, eps=1e-6):
        super().__init__(capacity, obs_dim)
        self.alpha = alpha
        self.eps = eps
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.max_priority = 1.0

    def add(self, obs, action, reward, next_obs, done):
        super().add(obs, action, reward, next_obs, done)
        self.priorities[(self.pos - 1) % self.capacity] = self.max_priority

    def sample(self, batch_size, beta=0.4):
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        idx = np.random.choice(self.size, size=batch_size, p=probs)
        weights = (self.size * probs[idx]) ** (-beta)
        weights /= weights.max()
        return (
            idx,
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
            weights.astype(np.float32),
        )

    def update_priorities(self, idx, priorities):
        priorities = np.asarray(priorities, dtype=np.float32)
        priorities = np.abs(priorities) + self.eps
        self.priorities[idx] = priorities
        self.max_priority = max(self.max_priority, float(priorities.max()))


class NStepBuffer:
    def __init__(self, n_steps, gamma):
        self.n_steps = n_steps
        self.gamma = gamma
        self.items = deque()

    def append(self, obs, action, reward, next_obs, done):
        self.items.append((obs, action, reward, next_obs, done))
        out = []
        if len(self.items) >= self.n_steps:
            out.append(self._build_transition())
        if done:
            while self.items:
                out.append(self._build_transition())
        return out

    def _build_transition(self):
        ret = 0.0
        next_obs = self.items[-1][3]
        done = self.items[-1][4]
        for i, (_, _, reward, candidate_next_obs, candidate_done) in enumerate(self.items):
            ret += (self.gamma ** i) * reward
            next_obs = candidate_next_obs
            done = candidate_done
            if candidate_done or i + 1 >= self.n_steps:
                break
        obs, action = self.items[0][0], self.items[0][1]
        self.items.popleft()
        return obs, action, ret, next_obs, done


def linear_schedule(start, end, progress):
    progress = min(max(progress, 0.0), 1.0)
    return start + progress * (end - start)


@torch.inference_mode()
def greedy_evaluate(model, n_episodes):
    device = next(model.parameters()).device
    env = make_env()
    was_training = model.training
    model.eval()

    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total = 0.0
        while True:
            q_values = model(torch.from_numpy(np.asarray(obs, dtype=np.float32))[None, :].to(device))
            action = int(q_values.argmax(dim=-1).item())
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
        returns.append(total)

    env.close()
    if was_training:
        model.train()
    return float(np.mean(returns)), float(np.std(returns))


@torch.inference_mode()
def a2c_actions(policy, observations, epsilon):
    device = next(policy.parameters()).device
    logits, _ = policy(torch.from_numpy(observations).to(device))
    actions = logits.argmax(dim=-1).cpu().numpy()
    if epsilon > 0:
        random_mask = np.random.random(size=len(actions)) < epsilon
        actions[random_mask] = np.random.randint(0, 4, size=random_mask.sum())
    return actions


@torch.inference_mode()
def dqn_actions(model, observations, epsilon):
    device = next(model.parameters()).device
    q_values = model(torch.from_numpy(observations).to(device))
    actions = q_values.argmax(dim=-1).cpu().numpy()
    if epsilon > 0:
        random_mask = np.random.random(size=len(actions)) < epsilon
        actions[random_mask] = np.random.randint(0, 4, size=random_mask.sum())
    return actions


def optimize(q_net, target_net, optimizer, replay, args, device, behavior_policy=None, step=0):
    beta_progress = step / max(1, args.total_steps)
    beta = linear_schedule(args.per_beta_start, 1.0, beta_progress)
    if isinstance(replay, PrioritizedReplayBuffer):
        idx, obs, actions, rewards, next_obs, dones, weights = replay.sample(args.batch_size, beta=beta)
    else:
        idx, obs, actions, rewards, next_obs, dones, weights = replay.sample(args.batch_size)

    obs = torch.from_numpy(obs).to(device)
    actions = torch.from_numpy(actions).to(device)
    rewards = torch.from_numpy(rewards).to(device) / args.reward_scale
    next_obs = torch.from_numpy(next_obs).to(device)
    dones = torch.from_numpy(dones).to(device)
    weights = torch.from_numpy(weights).to(device)

    q_values = q_net(obs).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        next_actions = q_net(next_obs).argmax(dim=1)
        next_q = target_net(next_obs).gather(1, next_actions[:, None]).squeeze(1)
        target = rewards + (args.gamma ** args.n_step) * (1.0 - dones) * next_q

    td_error = target - q_values
    dqn_loss = (F.smooth_l1_loss(q_values, target, reduction="none") * weights).mean()
    if behavior_policy is not None and args.bc_coef > 0:
        with torch.no_grad():
            behavior_logits, _ = behavior_policy(obs)
            behavior_actions = behavior_logits.argmax(dim=1)
        bc_coef = args.bc_coef
        if args.bc_decay_steps > 0:
            bc_coef *= max(0.0, 1.0 - step / args.bc_decay_steps)
        bc_loss = F.cross_entropy(q_net(obs), behavior_actions)
        loss = dqn_loss + bc_coef * bc_loss
    else:
        loss = dqn_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), args.max_grad_norm)
    optimizer.step()
    replay.update_priorities(idx, td_error.detach().abs().cpu().numpy())
    return float(loss.detach().cpu())


def pretrain_from_policy(q_net, policy, args, device):
    envs = [make_env() for _ in range(args.pretrain_envs)]
    observations = np.stack([env.reset(seed=args.seed + 10_000 + i)[0] for i, env in enumerate(envs)]).astype(np.float32)
    obs_chunks = []
    action_chunks = []

    collected = 0
    while collected < args.pretrain_steps:
        actions = a2c_actions(policy, observations, args.pretrain_epsilon)
        obs_chunks.append(observations.copy())
        action_chunks.append(actions.copy())
        collected += len(envs)

        next_observations = np.zeros_like(observations)
        for i, env in enumerate(envs):
            next_obs, _, terminated, truncated, _ = env.step(int(actions[i]))
            if terminated or truncated:
                next_obs, _ = env.reset()
            next_observations[i] = next_obs
        observations = next_observations

    for env in envs:
        env.close()

    obs = np.concatenate(obs_chunks, axis=0)[:args.pretrain_steps]
    actions = np.concatenate(action_chunks, axis=0)[:args.pretrain_steps]
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(obs), torch.from_numpy(actions).long())
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.pretrain_batch_size, shuffle=True)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=args.pretrain_lr)

    q_net.train()
    for epoch in range(args.pretrain_epochs):
        losses = []
        accuracy = []
        for obs_batch, action_batch in loader:
            obs_batch = obs_batch.to(device)
            action_batch = action_batch.to(device)
            q_values = q_net(obs_batch)
            loss = F.cross_entropy(q_values, action_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), args.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            accuracy.append(float((q_values.argmax(dim=1) == action_batch).float().mean().detach().cpu()))
        print(
            f"pretrain epoch {epoch + 1:2d}/{args.pretrain_epochs}  "
            f"ce {np.mean(losses):7.4f}  acc {np.mean(accuracy):5.3f}",
            flush=True,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--total_steps", type=int, default=700_000)
    p.add_argument("--n_envs", type=int, default=16)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--buffer_size", type=int, default=300_000)
    p.add_argument("--prioritized", action="store_true")
    p.add_argument("--per_alpha", type=float, default=0.6)
    p.add_argument("--per_beta_start", type=float, default=0.4)
    p.add_argument("--learning_starts", type=int, default=20_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--n_step", type=int, default=3)
    p.add_argument("--reward_scale", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--train_every", type=int, default=4)
    p.add_argument("--gradient_steps", type=int, default=1)
    p.add_argument("--target_update_every", type=int, default=4_000)
    p.add_argument("--max_grad_norm", type=float, default=10.0)
    p.add_argument("--eps_start", type=float, default=1.0)
    p.add_argument("--eps_end", type=float, default=0.03)
    p.add_argument("--eps_fraction", type=float, default=0.45)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=10_000)
    p.add_argument("--eval_every", type=int, default=50_000)
    p.add_argument("--eval_episodes", type=int, default=50)
    p.add_argument("--save", default="dqn_model.pt")
    p.add_argument("--best_save", default="dqn_model_best.pt")
    p.add_argument("--load", default=None)
    p.add_argument("--bootstrap_policy", default=None)
    p.add_argument("--bootstrap_steps", type=int, default=0)
    p.add_argument("--bootstrap_epsilon", type=float, default=0.15)
    p.add_argument("--bc_policy", default=None)
    p.add_argument("--bc_coef", type=float, default=0.0)
    p.add_argument("--bc_decay_steps", type=int, default=0)
    p.add_argument("--pretrain_policy", default=None)
    p.add_argument("--pretrain_steps", type=int, default=0)
    p.add_argument("--pretrain_envs", type=int, default=16)
    p.add_argument("--pretrain_epsilon", type=float, default=0.05)
    p.add_argument("--pretrain_epochs", type=int, default=4)
    p.add_argument("--pretrain_batch_size", type=int, default=512)
    p.add_argument("--pretrain_lr", type=float, default=3e-4)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}.", flush=True)

    envs = [make_env() for _ in range(args.n_envs)]
    observations = np.stack([env.reset(seed=args.seed + i)[0] for i, env in enumerate(envs)]).astype(np.float32)
    episode_returns = np.zeros(args.n_envs, dtype=np.float32)
    completed = []

    q_net = DQNModel(obs_dim=8, num_actions=4, hidden=args.hidden).to(device)
    target_net = DQNModel(obs_dim=8, num_actions=4, hidden=args.hidden).to(device)
    if args.load:
        q_net.load(args.load)
        print(f"loaded {args.load}", flush=True)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=args.lr)

    bootstrap_policy = None
    if args.bootstrap_policy:
        bootstrap_policy = RLModel(obs_dim=8, num_actions=4, hidden=256).to(device)
        bootstrap_policy.load(args.bootstrap_policy)
        bootstrap_policy.eval()
        print(f"bootstrapping behavior from {args.bootstrap_policy}", flush=True)

    bc_policy = None
    if args.bc_policy:
        bc_policy = RLModel(obs_dim=8, num_actions=4, hidden=256).to(device)
        bc_policy.load(args.bc_policy)
        bc_policy.eval()
        print(f"anchoring DQN updates to {args.bc_policy}", flush=True)

    if args.pretrain_policy and args.pretrain_steps > 0:
        pretrain_policy = RLModel(obs_dim=8, num_actions=4, hidden=256).to(device)
        pretrain_policy.load(args.pretrain_policy)
        pretrain_policy.eval()
        print(f"pretraining DQN argmax from {args.pretrain_policy}", flush=True)
        pretrain_from_policy(q_net, pretrain_policy, args, device)
        target_net.load_state_dict(q_net.state_dict())

    if args.prioritized:
        replay = PrioritizedReplayBuffer(args.buffer_size, obs_dim=8, alpha=args.per_alpha)
    else:
        replay = ReplayBuffer(args.buffer_size, obs_dim=8)
    n_step_buffers = [NStepBuffer(args.n_step, args.gamma) for _ in range(args.n_envs)]
    losses = deque(maxlen=500)
    best_eval = -float("inf")

    for step in range(1, args.total_steps + 1):
        if bootstrap_policy is not None and step <= args.bootstrap_steps:
            actions = a2c_actions(bootstrap_policy, observations, args.bootstrap_epsilon)
        else:
            progress = (step - args.bootstrap_steps) / max(1, args.total_steps * args.eps_fraction)
            epsilon = linear_schedule(args.eps_start, args.eps_end, progress)
            actions = dqn_actions(q_net, observations, epsilon)

        next_observations = np.zeros_like(observations)
        for i, env in enumerate(envs):
            next_obs, reward, terminated, truncated, _ = env.step(int(actions[i]))
            done = bool(terminated or truncated)
            episode_returns[i] += reward

            stored_next_obs = next_obs.astype(np.float32)
            for transition in n_step_buffers[i].append(
                observations[i].copy(),
                int(actions[i]),
                float(reward),
                stored_next_obs.copy(),
                done,
            ):
                replay.add(*transition)

            if done:
                completed.append(float(episode_returns[i]))
                episode_returns[i] = 0.0
                next_obs, _ = env.reset()
            next_observations[i] = next_obs
        observations = next_observations

        if replay.size >= args.learning_starts and step % args.train_every == 0:
            q_net.train()
            for _ in range(args.gradient_steps):
                losses.append(optimize(q_net, target_net, optimizer, replay, args, device, bc_policy, step))

        if step % args.target_update_every == 0:
            target_net.load_state_dict(q_net.state_dict())

        if step % args.log_every == 0:
            recent = completed[-50:] if completed else [0.0]
            loss = float(np.mean(losses)) if losses else 0.0
            print(
                f"step {step:8d}  replay {replay.size:7d}  loss {loss:7.4f}  "
                f"eps {locals().get('epsilon', args.bootstrap_epsilon):5.3f}  "
                f"return {np.mean(recent):7.2f} (last {len(recent)} eps)",
                flush=True,
            )

        if step % args.eval_every == 0:
            eval_mean, eval_std = greedy_evaluate(q_net, args.eval_episodes)
            print(f"eval {step:8d}  return {eval_mean:7.2f} +- {eval_std:6.2f}", flush=True)
            if eval_mean > best_eval:
                best_eval = eval_mean
                q_net.save(args.best_save)
                print(f"saved best {args.best_save} ({best_eval:.2f})", flush=True)

        if step % max(args.eval_every, args.log_every) == 0:
            q_net.save(args.save)

    q_net.save(args.save)
    if best_eval > -float("inf"):
        print(f"best eval {best_eval:.2f} saved to {args.best_save}", flush=True)
    print(f"saved {args.save}", flush=True)

    for env in envs:
        env.close()


if __name__ == "__main__":
    main()
