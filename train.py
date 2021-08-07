import numpy as np
import torch
import torch.multiprocessing as mp
import wandb
from gym_maze.envs.maze_env import MazeEnvSample5x5

from actor import Actor, Maze
from config import config
from embedding_model import EmbeddingModel, GFunction
from learner import Learner
from memory import Memory
from model import R2D2_agent57
from metacontroller import MetaController

def actor_process(actor_id, online_net, target_net, current_g_model, target_g_model, embedding_model, memory,
                 epsilon, lock):
    meta_controller = MetaController()
    actor = Actor(actor_id, online_net, target_net, current_g_model, target_g_model, embedding_model, memory, meta_controller,
                 epsilon, lock)
    actor.run()


def learner_process(online_net, target_net, current_g_model, target_g_model, embedding_model, memory, lock):
    leaner = Learner(online_net, target_net, current_g_model, target_g_model, embedding_model, memory, lock)
    leaner.run()

def main():
    n_actors = config.n_actors
    torch.manual_seed(config.random_seed)

    np.random.seed(config.random_seed)

    epsilon_list = [config.eps ** (1 + config.alpha * num / (n_actors - 1)) for num in range(1, n_actors + 1)]

    # wandb.init(project="ngu-maze", config=config.__dict__)
    env = Maze(MazeEnvSample5x5())
    # env = gym.make('CartPole-v1')
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n
    print("state size:", num_inputs)
    print("action size:", num_actions)

    # value net
    online_net = R2D2_agent57(num_inputs, num_actions)
    target_net = R2D2_agent57(num_inputs, num_actions)
    # online_net.R2D2_int.share_memory()
    # online_net.R2D2_ext.share_memory()
    # target_net.R2D2_int.share_memory()
    # target_net.R2D2_ext.share_memory()
    online_net.share_memory()
    target_net.share_memory()

    # RND model
    current_g_model = GFunction(obs_size=num_inputs)
    target_g_model = GFunction(obs_size=num_inputs)
    current_g_model.share_memory()
    target_g_model.share_memory()
    target_net.load_state_dict(online_net.state_dict())

    # Intrinsic reward 를 위해 controllable state 뽑아줌
    embedding_model = EmbeddingModel(obs_size=num_inputs, num_outputs=num_actions)
    embedding_model.share_memory()

    online_net.to(config.device)
    target_net.to(config.device)
    online_net.train()
    target_net.train()

    mp.Manager().register('Memory', Memory)
    manager = mp.Manager()
    memory = manager.Memory(config.replay_memory_capacity)

    lock = mp.Lock()

    # learner process
    processes = [mp.Process(
        target=learner_process,
        args=(online_net, target_net, current_g_model, target_g_model, embedding_model, memory, lock))]

    # actor processes
    for actor_id in range(n_actors):
        epsilon = epsilon_list[actor_id]
        processes.append(mp.Process(
            target=actor_process,
            args=(
            actor_id, online_net, target_net, current_g_model, target_g_model, embedding_model, memory, epsilon, lock)))

    for i in range(len(processes)):
        processes[i].start()

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
