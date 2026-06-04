import numpy as np

def softmax(x, beta):
    y = (np.exp(beta * x)) / (np.sum(np.exp(beta * x)))
    return y

class QLearningAgent:
    def __init__(self, alpha, beta, num_actions=2):
        # self.reward = reward    # 報酬
        # self.times = times      # 回数
        self.alpha = alpha      # 学習率
        self.beta = beta        # 逆温度
        self.num_actions = num_actions  # 選択肢の数
        self.q_values = np.zeros(num_actions)

    def reset(self):
        self.q_values = np.zeros(self.num_actions)

    def choose_action(self):
        action_probs = softmax(self.q_values, self.beta)
        action = np.random.choice(self.num_actions, p=action_probs)
        
        return action

    def update_q_value(self, action, reward):
        prediction_error = reward - self.q_values[action]
        self.q_values[action] += self.alpha * prediction_error