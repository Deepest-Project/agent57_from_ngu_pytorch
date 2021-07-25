import torch.optim as optim

from config import config
from model import R2D2

class Learner:
    def __init__(self, online_net, target_net, current_g_model, target_g_model, embedding_model, memory, lock):
        self.online_net = online_net
        self.target_net = target_net
        self.current_g_model = current_g_model
        self.target_g_model = target_g_model
        self.embedding_model = embedding_model
        self.memory = memory
        self.lock = lock
        self.optimizer = optim.Adam(online_net.parameters(), lr=config.lr)

        self.share_exp_mem = memory
        self.lock = lock

        self.steps = 0

    def run(self):
        while True:
            if len(self.share_exp_mem) > config.batch_size:
                batch, indexes, lengths = self.memory.sample(config.batch_size)

                for _ in range(5):
                    # todo : 1. congig.beta -> agent 별로. 2. gamma 추가
                    loss, td_error = R2D2.train_model(self.online_net, self.target_net, self.optimizer, batch,
                                                           lengths, config.beta)
                    if config.enable_ngu:
                        embedding_loss = self.embedding_model.train_model(batch)

                self.memory.update_priority(indexes, td_error, lengths)

                self.steps += 1
                if self.steps % config.update_target == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())
