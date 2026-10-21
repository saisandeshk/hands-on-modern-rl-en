"""Render trained DQN agent playing LunarLander-v3 and save as GIF."""

import argparse
from pathlib import Path

import imageio
import numpy as np
from stable_baselines3 import DQN


def render_episode(model, env_id="LunarLander-v3", max_steps=150, seed=None):
    import gymnasium as gym

    env = gym.make(env_id, render_mode="rgb_array")
    env.action_space.seed(seed if seed is not None else 0)
    state, _ = env.reset(seed=seed)

    frames = []
    total_reward = 0.0

    for _ in range(max_steps):
        frame = env.render()
        frames.append(frame)

        action, _ = model.predict(state, deterministic=True)
        state, reward, terminated, truncated, _ = env.step(int(action))
        total_reward += reward

        if terminated or truncated:
            break

    env.close()
    return frames, total_reward


def render_smoke_random(args):
    """Smoke path: no model needed, random actions, writes under runs/."""
    import gymnasium as gym

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env.action_space.seed(args.seed)
    state, _ = env.reset(seed=args.seed)
    frames = []
    for _ in range(args.max_steps):
        frames.append(env.render())
        state, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            break
    env.close()
    imageio.mimsave(output_path, frames, duration=1000 / args.fps, loop=0)
    print(f"Saved smoke render to {output_path} ({len(frames)} frames)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--output", type=str, default="output/lunarlander_play.gif")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke-test mode: random policy, few steps, writes only under runs/",
    )
    args = parser.parse_args()

    if args.smoke:
        args.output = "runs/smoke_render_lunarlander/lunarlander_smoke.gif"
        args.episodes = 1
        args.max_steps = 20
        args.seeds = [args.seed]
        render_smoke_random(args)
        return

    if args.model is None:
        raise SystemExit("--model is required without --smoke")
    model = DQN.load(args.model)
    all_frames = []

    seeds = args.seeds or [None] * args.episodes

    for ep, seed in enumerate(seeds[: args.episodes]):
        frames, reward = render_episode(model, max_steps=args.max_steps, seed=seed)
        seed_label = f", seed={seed}" if seed is not None else ""
        print(
            f"Episode {ep + 1}: reward={reward:.1f}, frames={len(frames)}{seed_label}"
        )
        all_frames.extend(frames)
        # Add a few black frames as separator between episodes
        if ep < args.episodes - 1:
            black = np.zeros_like(frames[0])
            all_frames.extend([black] * 5)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, all_frames, duration=1000 / args.fps, loop=0)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
