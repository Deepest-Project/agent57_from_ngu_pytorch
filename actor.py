import numpy as np
import torch
from gym import Wrapper
from gym_maze.envs.maze_env import MazeEnvSample5x5

import wandb
from config import config
from embedding_model import compute_intrinsic_reward
from memory import LocalBuffer


# todo : MetaController 구현
class Maze(Wrapper):
    def step(self, action: int):
        obs, rew, done, info = super().step(["N", "E", "S", "W"][action])
        self.set.add((obs[0], obs[1]))
        if rew > 0:
            rew = 10
        return obs / 10, rew, done, info

    def reset(self):
        self.set = set()
        return super().reset()


def get_action(state, target_net, epsilon, env, hidden, beta):
    action, hidden = target_net.get_action(state, hidden, beta)

    if np.random.rand() <= epsilon:
        return env.action_space.sample(), hidden
    else:
        return action, hidden

import gym

class Actor:
    def __init__(self, actor_id, online_net, target_net, current_g_model, target_g_model, embedding_model, memory,
                 epsilon, lock):

        self.env = gym.make('CartPole-v1') #gym.Maze(MazeEnvSample5x5())
        self.env.seed(config.random_seed)
        self.env.action_space.seed(config.random_seed)

        self.actor_id = actor_id
        self.online_net = online_net
        self.target_net = target_net
        self.current_g_model = current_g_model
        self.target_g_model = target_g_model
        self.embedding_model = embedding_model
        self.memory = memory
        self.epsilon = epsilon
        self.lock = lock

    def run(self):
        steps = 0
        loss = 0
        local_buffer = LocalBuffer()
        sum_reward = 0
        sum_augmented_reward = 0
        sum_obs_set = 0

        for episode in range(30000):
            done = False
            state = self.env.reset()
            state = torch.Tensor(state).to(config.device)

            hidden = (
                torch.Tensor().new_zeros(1, 1, config.hidden_size),
                torch.Tensor().new_zeros(1, 1, config.hidden_size),
            )

            episodic_memory = [self.embedding_model.embedding(state)]

            episode_steps = 0
            horizon = 100
            MA = 0

            while not done:
                steps += 1
                episode_steps += 1

                action, new_hidden = get_action(state, self.target_net, self.epsilon, self.env, hidden,
                                                beta=config.beta)
                print(f"self.actor_id={self.actor_id}, action={action}")

                next_state, env_reward, done, _ = self.env.step(action)
                next_state = torch.Tensor(next_state)

                augmented_reward = env_reward
                if config.enable_ngu:
                    next_state_emb = self.embedding_model.embedding(next_state)

                    c_out = self.current_g_model(next_state)
                    alpha = self.target_g_model.train_model(c_out, next_state)

                    intrinsic_reward, MA = compute_intrinsic_reward(episodic_memory, next_state_emb, alpha=alpha,
                                                                    episode_steps=episode_steps, MA=MA)

                    episodic_memory.append(next_state_emb)
                    beta = config.beta
                    augmented_reward = env_reward + beta * intrinsic_reward

                mask = 0 if done else 1

                local_buffer.push(state, next_state, action, augmented_reward, mask, hidden)
                hidden = new_hidden

                # todo :get_td_error 할 때 config.beta가 아니라 beta 가변적으로 받을 수 있도록
                if len(local_buffer.memory) == config.local_mini_batch:
                    batch, lengths = local_buffer.sample()
                    td_error = self.R2D2.get_td_error(self.online_net, self.target_net, batch, lengths, config.beta)
                    self.memory.push(td_error, batch, lengths)

                sum_reward += env_reward
                state = next_state
                sum_augmented_reward += augmented_reward

            # if episode > 0 and episode % config.log_interval == 0:
            #     mean_reward = sum_reward / config.log_interval
            #     mean_augmented_reward = sum_augmented_reward / config.log_interval
            #     metrics = {
            #         "episode": episode,
            #         "mean_reward": mean_reward,
            #         "epsilon": self.epsilon,
            #         # "embedding_loss": embedding_loss,
            #         "loss": loss,
            #         "mean_augmented_reward": mean_augmented_reward,
            #         "steps": steps,
            #         "sum_obs_set": sum_obs_set / config.log_interval,
            #     }
            #     print(metrics)
            #     wandb.log(metrics)
            #
            #     sum_reward = 0
            #     sum_augmented_reward = 0
            #     sum_obs_set = 0
