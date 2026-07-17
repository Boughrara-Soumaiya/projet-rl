# train.py
import numpy as np
import pickle
from environment import TrafficGridEnv
from agent import QLearningAgent

def run_fixed_cycle_baseline(steps=200, cycle_length=5):
    """
    Simule une régulation classique par cycle fixe (non adaptatif).
    Alterne les feux toutes les 'cycle_length' étapes.

    Retourne la congestion moyenne PAR AGENT (véhicules/axe), pour être
    directement comparable à 'avg_queue' retourné par train_agents.
    """
    env = TrafficGridEnv()
    states = env.reset()
    total_waiting_cars = 0
    
    current_action = 0
    for step in range(steps):
        # Alterne l'action de chaque agent périodiquement
        if step % cycle_length == 0:
            current_action = 1 - current_action
        
        actions = [current_action] * env.num_agents
        next_states, rewards, _, _ = env.step(actions)
        
        # Métrique : somme globale des files d'attente à cet instant
        total_waiting_cars += np.sum(env.queues)
        
    # Normalisation par agent pour être comparable à train_agents (même unité)
    return total_waiting_cars / (steps * env.num_agents)

def train_agents(episodes=1000, steps_per_episode=100):
    """
    Entraîne les agents Q-learning indépendants (IQL).
    """
    env = TrafficGridEnv()
    # Initialisation d'un agent par intersection
    agents = [QLearningAgent() for _ in range(env.num_agents)]
    
    history = []

    for episode in range(episodes):
        states = env.reset()
        episode_reward = 0
        
        for step in range(steps_per_episode):
            # Chaque agent choisit son action indépendamment
            actions = [agents[i].choose_action(states[i]) for i in range(env.num_agents)]
            
            # Transition de l'environnement
            next_states, rewards, _, _ = env.step(actions)
            
            # Apprentissage local pour chaque agent
            for i in range(env.num_agents):
                agents[i].learn(states[i], actions[i], rewards[i], next_states[i])
                
            states = next_states
            episode_reward += sum(rewards)
            
        # Diminution de l'exploration (Epsilon-decay)
        for agent in agents:
            agent.decay_epsilon()
            
        # Log des performances tous les 50 épisodes
        if (episode + 1) % 50 == 0:
            avg_queue = -episode_reward / (steps_per_episode * env.num_agents)
            print(f"Épisode {episode+1}/{episodes} | Congestion moyenne : {avg_queue:.2f} voitures/axe | Epsilon: {agents[0].epsilon:.3f}")
            history.append(avg_queue)

    # Sauvegarde des agents entraînés
    with open("trained_agents.pkl", "wb") as f:
        pickle.dump(agents, f)
    print("\n[SUCCÈS] Agents entraînés et sauvegardés sous 'trained_agents.pkl'")
    return history

if __name__ == "__main__":
    print("--- 1. ÉVALUATION DE LA BASELINE (FEUX À CYCLE FIXE) ---")
    baseline_perf = run_fixed_cycle_baseline(steps=500, cycle_length=5)
    print(f"Congestion moyenne par agent avec cycle fixe : {baseline_perf:.2f} véhicules/axe\n")
    
    print("--- 2. ENTRAÎNEMENT DES AGENTS MARL ---")
    history = train_agents(episodes=800, steps_per_episode=100)

    print("\n--- 3. COMPARAISON (même unité : véhicules/axe/agent) ---")
    print(f"Baseline (cycle fixe)      : {baseline_perf:.2f}")
    print(f"MARL (Q-learning, fin)     : {history[-1]:.2f}")
    gain = 100 * (baseline_perf - history[-1]) / baseline_perf
    print(f"Réduction de la congestion : {gain:.1f}%")