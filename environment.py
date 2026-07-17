# environment.py
import numpy as np

class TrafficGridEnv:
    def __init__(self, grid_size=(2, 2), max_queue=10):
        self.grid_size = grid_size
        self.num_agents = grid_size[0] * grid_size[1]
        self.max_queue = max_queue
        
        # Directions: 0=North, 1=South, 2=East, 3=West
        self.directions = ['N', 'S', 'E', 'W']
        
        # Mapping de l'index de l'agent (0 à 3) vers ses coordonnées (x, y)
        self.agent_coords = [(i, j) for i in range(grid_size[0]) for j in range(grid_size[1])]
        
        self.reset()

    def reset(self):
        # File d'attente pour chaque agent dans les 4 directions
        # queues[agent_id][dir_idx] = nombre de voitures
        self.queues = np.zeros((self.num_agents, 4), dtype=int)
        
        # État initial des feux (0: Vert NS / Rouge EO, 1: Rouge NS / Vert EO)
        self.lights = np.zeros(self.num_agents, dtype=int)
        return self._get_states()

    def _get_states(self):
        """
        Retourne l'état discrétisé pour chaque agent.
        Chaque état d'un agent est un tuple de 4 valeurs (N, S, E, W) discrétisées.
        """
        discretized_states = []
        for agent_id in range(self.num_agents):
            state = []
            for q_len in self.queues[agent_id]:
                # Discrétisation simple : 0, 1-2 (faible), 3-5 (moyen), >5 (fort)
                if q_len == 0:
                    state.append(0)
                elif q_len <= 2:
                    state.append(1)
                elif q_len <= 5:
                    state.append(2)
                else:
                    state.append(3)
            discretized_states.append(tuple(state))
        return discretized_states

    def step(self, actions):
        """
        Exécute les actions des agents.
        actions: liste de taille num_agents contenant 0 (NS) ou 1 (EO)
        """
        rewards = []
        old_queues_sum = np.sum(self.queues)
        
        # 1. Appliquer les feux
        self.lights = np.array(actions)
        
        # 2. Déplacement des véhicules
        for agent_id in range(self.num_agents):
            x, y = self.agent_coords[agent_id]
            
            # Traiter l'axe Nord-Sud (Action = 0 -> Vert)
            if self.lights[agent_id] == 0:
                # Nord s'écoule vers le Sud (y+1)
                if self.queues[agent_id][0] > 0:
                    self.queues[agent_id][0] -= 1
                    if y + 1 < self.grid_size[1]: # Reste dans la grille
                        neighbor_id = self._get_agent_id(x, y + 1)
                        self.queues[neighbor_id][0] = min(self.queues[neighbor_id][0] + 1, self.max_queue)
                # Sud s'écoule vers le Nord (y-1)
                if self.queues[agent_id][1] > 0:
                    self.queues[agent_id][1] -= 1
                    if y - 1 >= 0:
                        neighbor_id = self._get_agent_id(x, y - 1)
                        self.queues[neighbor_id][1] = min(self.queues[neighbor_id][1] + 1, self.max_queue)
                        
            # Traiter l'axe Est-Ouest (Action = 1 -> Vert)
            else:
                # Est s'écoule vers l'Ouest (x-1)
                if self.queues[agent_id][2] > 0:
                    self.queues[agent_id][2] -= 1
                    if x - 1 >= 0:
                        neighbor_id = self._get_agent_id(x - 1, y)
                        self.queues[neighbor_id][2] = min(self.queues[neighbor_id][2] + 1, self.max_queue)
                # Ouest s'écoule vers l'Est (x+1)
                if self.queues[agent_id][3] > 0:
                    self.queues[agent_id][3] -= 1
                    if x + 1 < self.grid_size[0]:
                        neighbor_id = self._get_agent_id(x + 1, y)
                        self.queues[neighbor_id][3] = min(self.queues[neighbor_id][3] + 1, self.max_queue)

        # 3. Arrivées aléatoires de véhicules aux frontières de la grille
        for agent_id in range(self.num_agents):
            x, y = self.agent_coords[agent_id]
            # Si l'agent est sur une bordure, du trafic externe arrive aléatoirement
            if y == 0 and np.random.rand() < 0.4: # Entrée Nord
                self.queues[agent_id][0] = min(self.queues[agent_id][0] + 1, self.max_queue)
            if y == self.grid_size[1]-1 and np.random.rand() < 0.4: # Entrée Sud
                self.queues[agent_id][1] = min(self.queues[agent_id][1] + 1, self.max_queue)
            if x == self.grid_size[0]-1 and np.random.rand() < 0.4: # Entrée Est
                self.queues[agent_id][2] = min(self.queues[agent_id][2] + 1, self.max_queue)
            if x == 0 and np.random.rand() < 0.4: # Entrée Ouest
                self.queues[agent_id][3] = min(self.queues[agent_id][3] + 1, self.max_queue)

        # 4. Calcul des récompenses (Négatif de la somme des files d'attente locales)
        # On veut minimiser la congestion globale = maximiser -somme(queues)
        for agent_id in range(self.num_agents):
            local_congestion = np.sum(self.queues[agent_id])
            rewards.append(-float(local_congestion))
            
        return self._get_states(), rewards, False, {}

    def _get_agent_id(self, x, y):
        return x * self.grid_size[1] + y