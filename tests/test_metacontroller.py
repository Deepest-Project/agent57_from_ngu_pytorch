import numpy as np

def sigmoid(x):
    return 1 / (1+np.exp(-x))

from collections import deque
# policy_num == N_agents ? => No.
from metacontroller import MetaController

# policy_num 이 뭐지 ? => actor num
#

if __name__ == '__main__':
    """
    class Actor:
    def __init(self, gc)__:
        self.metacontroller = gc

    def get_beta_gamma(self, current_episode):
        self.beta_gamma = self.metacontroller.get_beta_gamma(current_episode)
        """
    gcm = MetaController()
    actor = Actor(gcm)

    for eps in episode:
        actor.mc.reset(episode_reward)
