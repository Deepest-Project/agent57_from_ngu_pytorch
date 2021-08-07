import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


from collections import deque


# policy_num == N_agents ? => No.
class MetaController:
    def __init__(self, ucb_window_size=90, policy_num=32, ucb_beta=1, ucb_epsilon=0.5, training=False):

        self.training = training
        self.ucb_data = []  # For history storage
        # self.ucb_window_size = ucb_window_size
        self.policy_num = policy_num  # Total number of policies
        self.ucb_beta = ucb_beta  # UCB β
        self.ucb_epsilon = ucb_epsilon  # Choose to use UCB with ε
        self.policy_index = np.random.randint(self.policy_num)

        self.episode_count = 0
        self.ucb_data = deque([], maxlen=ucb_window_size)

        # use sliding-window UCB(UpperConfidenceBandit)
        self.beta_list = self.create_beta_list()
        self.gamma_list = self.create_gamma_list_agent57()
        # self.beta = self.beta_list[self.policy_index]
        self.beta = 0#np.zeros([1])
        self.gamma = self.gamma_list[self.policy_index]

    def create_beta_list(self, max_beta=0.3):
        beta_list = []
        for i in range(self.policy_num):
            if i == 0:
                b = 0
            elif i == self.policy_num - 1:
                b = max_beta
            else:
                b = 10 * (2 * i - (self.policy_num - 2)) / (self.policy_num - 2)
                b = max_beta * sigmoid(b)
            beta_list.append(b)
        return beta_list

    def create_gamma_list_agent57(self, gamma0=0.9999, gamma1=0.997, gamma2=0.99):
        gamma_list = []
        for i in range(self.policy_num):
            if i == 1:
                g = gamma0
            elif 2 <= i and i <= 6:
                g = 10 * ((2 * i - 6) / 6)
                g = gamma1 + (gamma0 - gamma1) * sigmoid(g)
            elif i == 7:
                g = gamma1
            else:
                g = (self.policy_num - 9) * np.log(1 - gamma1) + (i - 8) * np.log(1 - gamma2)
                g /= self.policy_num - 9
                g = 1 - np.exp(g)
            gamma_list.append(g)
        return gamma_list

    def get_beta_gamma(self):
        return self.beta_list[self.pocicy_index], self.gamma_list[self.pocicy_index]

    def reset_states(self, episode_reward):
        # if not training, set to policy 0 => no search
        if not self.training:
            policy_index = 0
        else:
            if self.episode_count > 0:
                self.ucb_data.append([self.policy_index, episode_reward])

            if self.episode_count < self.policy_num:
                policy_index = self.episode_count
            else:
                r = np.random.random()
                if r < self.ucb_epsilon:
                    # ランダムでpolicyを決定
                    policy_index = np.random.randint(self.policy_num)  # a <= n <= b
                else:
                    # calculate UCB
                    N = [1 for _ in range(self.policy_num)]
                    u = [0 for _ in range(self.policy_num)]
                    for d in self.ucb_data:
                        N[d[0]] += 1
                        u[d[0]] += d[1]

                    for i in range(self.policy_num):
                        u[i] /= N[i]

                    count = len(self.ucb_data)
                    k = [0 for _ in range(self.policy_num)]
                    for i in range(self.policy_num):
                        k[i] = u[i] + self.ucb_beta * np.sqrt(np.log(count) / N[i])

                    self.policy_index = np.argmax(k)

            self.episode_count += 1
            # self.beta, self.gamma = self.get_beta_gamma()


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
