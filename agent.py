# agent.py
import numpy as np

class QLearningAgent:
    def __init__(self, action_space_dim=2, lr=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.action_space_dim = action_space_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Table Q : clé = tuple d'état (N, S, E, W), valeur = tableau d'utilité des actions [Q(s,0), Q(s,1)]
        self.q_table = {}

    def get_q_values(self, state):
        if state not in self.q_table:
            # Initialisation optimiste ou à zéro
            self.q_table[state] = np.zeros(self.action_space_dim)
        return self.q_table[state]

    def choose_action(self, state):
        # Politique Epsilon-Greedy
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space_dim)
        
        q_values = self.get_q_values(state)
        # Ajout d'un léger bruit pour casser les égalités de manière aléatoire
        return np.argmax(q_values + np.random.randn(self.action_space_dim) * 1e-5)

    def learn(self, state, action, reward, next_state):
        q_current = self.get_q_values(state)[action]
        q_next_max = np.max(self.get_q_values(next_state))
        
        # Règle de mise à jour Bellman temporelle standard
        new_q = q_current + self.lr * (reward + self.gamma * q_next_max - q_current)
        self.q_table[state][action] = new_q

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        