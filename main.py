import gym

import torch
import numpy as np
import argparse

from sac_agent import *
from utils import *

def run_agent(args):
    env = gym.make(args.env_name)
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # env.seed(args.seed)

    agent = SACAgent(env.observation_space.shape[0], env.action_space.shape[0], args)
    memory = ReplayMemory(args.replay_size)

    total_steps = 0
    updates = 0

    for e in range(args.total_episodes):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        while not done:
            if args.start_steps > total_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
        
            if len(memory) > args.batch_size:
                for _ in range(args.updates_per_step):
                    critic0_loss, critic1_loss, policy_loss, ent_loss = agent.update_params(memory, args.batch_size, updates)
                    updates += 1
            
            next_state, reward, done, _ = env.step(action)
            episode_steps += 1
            episode_reward += reward

            total_steps += 1
            mask = 1. if episode_steps == env._max_episode_steps else float(not done)

            memory.push(state, action, next_state, reward, done)
            state = next_state
    
        print('episode: {}, episode steps: {}, episode reward: {}'.format(e, episode_steps, episode_reward))

        if e % 10 == 0 and args.eval == True:
            episode_reward_e = 0
            for _ in range(10):
                state = env.reset()
                done = False
                while not done:
                    action = agent.select_action(state, eval=True)
                    next_state, reward, done, _ = env.step(action)
                    episode_reward_e += reward
                    state = next_state
            avg_reward_e = episode_reward_e / 10
            print('test avg reward: {}'.format(avg_reward_e))
        
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default='HalfCheetah-v2')
    parser.add_argument('--policy', default='Gaussian')
    parser.add_argument('--eval', type=bool, default=True)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--auto_entropy_tuning', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=456)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--total_episodes', type=int, default=1000)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--updates_per_step', type=int, default=1)
    parser.add_argument('--start_steps', type=int, default=10000)
    parser.add_argument('--target_update_interval', type=int, default=1)
    parser.add_argument('--replay_size', type=int, default=10000000)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()

    run_agent(args)