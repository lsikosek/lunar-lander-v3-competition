import argparse
import numpy as np
import torch
import torch.nn.functional as F

from env import make_env
from rl_model import RLModel


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
            obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32))[None, :].to(device)
            logits, _ = model(obs_t)
            action = int(logits.argmax(dim=-1).item())
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
        returns.append(total)

    env.close()
    if was_training:
        model.train()
    return float(np.mean(returns)), float(np.std(returns))


def train_ppo_update(model, optimizer, rollout, args):
    device = next(model.parameters()).device

    obs = torch.from_numpy(rollout["observations"]).to(device)
    actions = torch.from_numpy(rollout["actions"]).long().to(device)
    raw_rewards = torch.from_numpy(rollout["rewards"]).to(device)
    is_done = torch.from_numpy(rollout["is_done"]).to(device)
    old_logits = torch.from_numpy(rollout["logits"]).to(device)
    last_obs = torch.from_numpy(rollout["last_obs"]).to(device)
    if args.x_penalty or args.vx_penalty or args.angle_penalty:
        raw_rewards = raw_rewards - args.x_penalty * obs[..., 0].abs()
        raw_rewards = raw_rewards - args.vx_penalty * obs[..., 2].abs()
        raw_rewards = raw_rewards - args.angle_penalty * obs[..., 4].abs()
    rewards = raw_rewards / model.reward_scale

    B, T = actions.shape
    with torch.no_grad():
        _, old_values = model(obs)
        _, last_value = model(last_obs)

        advantages = torch.empty_like(rewards)
        future_advantage = torch.zeros(B, device=device)
        for t in range(T - 1, -1, -1):
            next_value = last_value if t == T - 1 else old_values[:, t + 1]
            next_is_live = 1.0 - is_done[:, t]
            delta = rewards[:, t] + model.gamma * next_value * next_is_live - old_values[:, t]
            future_advantage = delta + model.gamma * args.gae_lambda * next_is_live * future_advantage
            advantages[:, t] = future_advantage

        returns = advantages + old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        old_log_probs = F.log_softmax(old_logits, dim=-1)
        old_action_log_probs = old_log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

    obs_flat = obs.reshape(B * T, model.obs_dim)
    actions_flat = actions.reshape(B * T)
    returns_flat = returns.reshape(B * T)
    advantages_flat = advantages.reshape(B * T)
    old_action_log_probs_flat = old_action_log_probs.reshape(B * T)

    n_samples = obs_flat.shape[0]
    minibatch_size = n_samples if args.minibatch_size <= 0 else min(args.minibatch_size, n_samples)

    diagnostics = {
        "loss": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
    }

    model.train()
    for _ in range(args.ppo_epochs):
        permutation = torch.randperm(n_samples, device=device)
        for start in range(0, n_samples, minibatch_size):
            idx = permutation[start:start + minibatch_size]

            logits, values = model(obs_flat[idx])
            log_probs = F.log_softmax(logits, dim=-1)
            action_log_probs = log_probs.gather(-1, actions_flat[idx, None]).squeeze(-1)
            entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()

            ratio = (action_log_probs - old_action_log_probs_flat[idx]).exp()
            unclipped = ratio * advantages_flat[idx]
            clipped = torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef) * advantages_flat[idx]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, returns_flat[idx])
            loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            diagnostics["loss"].append(float(loss.detach().cpu()))
            diagnostics["policy_loss"].append(float(policy_loss.detach().cpu()))
            diagnostics["value_loss"].append(float(value_loss.detach().cpu()))
            diagnostics["entropy"].append(float(entropy.detach().cpu()))

    return {
        key: torch.tensor(np.mean(values), device=device)
        for key, values in diagnostics.items()
    } | {"mean_return": returns.mean().detach()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["a2c", "ppo"], default="a2c")
    p.add_argument("--n_envs", type=int, default=16) # ok
    p.add_argument("--n_steps", type=int, default=100) # ok
    p.add_argument("--n_iterations", type=int, default=5000) # ok
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--entropy_coef", type=float, default=0.001)
    p.add_argument("--value_coef", type=float, default=0.5)
    p.add_argument("--reward_scale", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=10) # ok
    p.add_argument("--save_every", type=int, default=50) # ok
    p.add_argument("--eval_every", type=int, default=0)
    p.add_argument("--eval_episodes", type=int, default=50)
    p.add_argument("--ppo_epochs", type=int, default=4)
    p.add_argument("--minibatch_size", type=int, default=512)
    p.add_argument("--clip_coef", type=float, default=0.2)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--x_penalty", type=float, default=0.0)
    p.add_argument("--vx_penalty", type=float, default=0.0)
    p.add_argument("--angle_penalty", type=float, default=0.0)
    p.add_argument("--load", default=None)
    p.add_argument("--best_save", default=None)
    p.add_argument("--save", default="model.pt")
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    envs = [make_env() for _ in range(args.n_envs)]

    model = RLModel(obs_dim=8, num_actions=4, hidden=args.hidden,
                    reward_scale=args.reward_scale, gamma=args.gamma)
    if args.load:
        model.load(args.load)
        print(f"loaded {args.load}")

    if torch.cuda.is_available():
        print("Using CUDA.")
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ep_return = np.zeros(args.n_envs, dtype=np.float32)
    completed = []
    best_eval = -float("inf")

    for it in range(args.n_iterations):
        rollout = model.collect_rollouts(envs, args.n_steps)

        for t in range(args.n_steps):
            ep_return += rollout["rewards"][:, t]
            for i in range(args.n_envs):
                if rollout["is_done"][i, t]:
                    completed.append(float(ep_return[i]))
                    ep_return[i] = 0.0

        if args.algo == "ppo":
            losses = train_ppo_update(model, optimizer, rollout, args)
        else:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = model.loss_a2c(rollout, entropy_coef=args.entropy_coef, value_coef=args.value_coef)
            losses["loss"].backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

        if it % args.log_every == 0:
            recent = completed[-50:] if completed else [0.0]
            print(f"iter {it:5d}  loss {losses['loss'].item():7.3f}  "
                  f"pi {losses['policy_loss'].item():6.3f}  v {losses['value_loss'].item():6.3f}  "
                  f"H {losses['entropy'].item():5.3f}  "
                  f"return {np.mean(recent):7.2f} (last {len(recent)} eps)", flush=True)

        if it > 0 and it % args.save_every == 0:
            model.save(args.save)

        if args.eval_every > 0 and it > 0 and it % args.eval_every == 0:
            eval_mean, eval_std = greedy_evaluate(model, args.eval_episodes)
            print(f"eval {it:5d}  return {eval_mean:7.2f} +- {eval_std:6.2f} "
                  f"({args.eval_episodes} eps)", flush=True)
            if args.best_save and eval_mean > best_eval:
                best_eval = eval_mean
                model.save(args.best_save)
                print(f"saved best {args.best_save} ({best_eval:.2f})", flush=True)

    model.save(args.save)
    print(f"saved {args.save}")
    if args.best_save and best_eval > -float("inf"):
        print(f"best eval {best_eval:.2f} saved to {args.best_save}")

    for env in envs:
        env.close()


if __name__ == "__main__":
    main()
