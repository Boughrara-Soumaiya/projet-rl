# app.py
import time
import pickle

import numpy as np
import pandas as pd
import streamlit as st

from environment import TrafficGridEnv
from agent import QLearningAgent
from train import run_fixed_cycle_baseline

st.set_page_config(page_title="Régulation de trafic par agents Q-learning", layout="wide")

ACCENT = "#B8503F"
GO = "#4C7A51"

st.title("Régulation de trafic par agents Q-learning")
st.caption(
    "Grille 2×2 d'intersections. Chaque intersection est un agent indépendant : "
    "il ne voit que ses propres files d'attente et choisit seul son cycle de feux, "
    "sans coordination centrale."
)

# ---------------------------------------------------------------------------
# Chargement des agents
# ---------------------------------------------------------------------------
@st.cache_resource
def load_agents():
    try:
        with open("trained_agents.pkl", "rb") as f:
            return pickle.load(f), True
    except FileNotFoundError:
        env = TrafficGridEnv()
        return [QLearningAgent(epsilon=0.1) for _ in range(env.num_agents)], False


@st.cache_data
def get_baseline_reference():
    return run_fixed_cycle_baseline(steps=500, cycle_length=5)


agents, model_loaded = load_agents()
baseline_reference = get_baseline_reference()

# ---------------------------------------------------------------------------
# Barre latérale — réglages
# ---------------------------------------------------------------------------
st.sidebar.header("Réglages")

if model_loaded:
    st.sidebar.success("Agents entraînés chargés")
else:
    st.sidebar.warning("Aucun modèle entraîné trouvé — agents non entraînés")

mode = st.sidebar.radio("Régulation", ["Agents Q-learning", "Cycle fixe (référence)"])
max_lane_capacity = st.sidebar.slider("Capacité par voie", 5, 20, 10)
sim_steps = st.sidebar.slider("Durée de la simulation (pas de temps)", 50, 500, 150)
speed = st.sidebar.slider("Vitesse d'affichage", 0.02, 0.5, 0.1)
run_sim = st.sidebar.button("Lancer la simulation", type="primary")

with st.sidebar.expander("Réglages avancés de l'agent"):
    exploration_rate = st.slider("Exploration (epsilon)", 0.0, 1.0, 0.0 if model_loaded else 0.2, step=0.05)
    learning_rate_alpha = st.slider("Taux d'apprentissage (alpha)", 0.01, 0.5, 0.1, step=0.01)

for agent in agents:
    agent.epsilon = exploration_rate
    agent.lr = learning_rate_alpha

env = TrafficGridEnv(max_queue=max_lane_capacity)
states = env.reset()

# ---------------------------------------------------------------------------
# Zone principale
# ---------------------------------------------------------------------------
col_grid, col_metrics = st.columns([1.3, 1])

with col_grid:
    st.subheader("Grille")
    grid_placeholder = st.empty()

with col_metrics:
    st.subheader("Congestion")
    metric_placeholder = st.empty()
    chart_placeholder = st.empty()

diagnostic = st.expander("Diagnostic avancé (couverture de l'espace d'état, convergence)")
with diagnostic:
    d_col1, d_col2 = st.columns(2)
    coverage_placeholder = d_col1.empty()
    q_chart_placeholder = d_col2.empty()

with st.expander("Comment ça marche ?"):
    st.markdown("""
Chaque intersection est un agent Q-learning indépendant : il observe ses propres
files d'attente (discrétisées en 4 niveaux par axe) et choisit entre deux actions —
feu vert Nord-Sud ou Est-Ouest. Les agents ne communiquent pas entre eux ; chacun
traite le comportement de ses voisins comme faisant partie de l'incertitude de
l'environnement.

**Règle de mise à jour (Bellman) :**
""")
    st.latex(r"Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]")
    st.markdown(
        f"Référence : une régulation à cycle fixe (sans apprentissage) atteint une "
        f"congestion moyenne de **{baseline_reference:.2f} véhicules par axe et par agent** "
        f"sur 500 pas de simulation."
    )


def render_grid(queues, lights):
    rows = []
    for i in range(4):
        light_state = "N-S vert" if lights[i] == 0 else "E-O vert"
        rows.append({
            "Intersection": i,
            "Nord": queues[i][0], "Sud": queues[i][1],
            "Est": queues[i][2], "Ouest": queues[i][3],
            "Feu": light_state,
        })
    return pd.DataFrame(rows)


if run_sim:
    congestion_history = []
    agent_states_visited = [set() for _ in range(env.num_agents)]
    q_sample_history = []
    target_state = (3, 3, 3, 3)

    np.random.seed(42)

    for t in range(sim_steps):
        if mode == "Agents Q-learning":
            actions = [agents[i].choose_action(states[i]) for i in range(env.num_agents)]
        else:
            cycle_val = 0 if (t // 5) % 2 == 0 else 1
            actions = [cycle_val] * env.num_agents

        next_states, rewards, _, _ = env.step(actions)

        for i in range(env.num_agents):
            agent_states_visited[i].add(states[i])

        q_sample_history.append(agents[0].get_q_values(target_state)[0])
        states = next_states

        current_congestion = int(np.sum(env.queues))
        congestion_history.append(current_congestion)
        avg_per_agent = np.mean(congestion_history) / env.num_agents
        delta_vs_baseline = 100 * (baseline_reference - avg_per_agent) / baseline_reference

        grid_placeholder.dataframe(render_grid(env.queues, env.lights), hide_index=True, width="stretch")

        with metric_placeholder.container():
            m1, m2 = st.columns(2)
            m1.metric("Véhicules en attente", current_congestion)
            m2.metric(
                "Moyenne / axe / agent", f"{avg_per_agent:.1f}",
                delta=f"{delta_vs_baseline:+.0f}% vs. cycle fixe",
                delta_color="normal",
            )

        chart_placeholder.line_chart(
            pd.DataFrame({"véhicules bloqués": congestion_history}), height=220
        )

        with coverage_placeholder.container():
            coverage_data = [{
                "Agent": i,
                "États découverts": len(agent_states_visited[i]),
                "Couverture": f"{len(agent_states_visited[i]) / 256 * 100:.1f}%",
            } for i in range(env.num_agents)]
            st.dataframe(pd.DataFrame(coverage_data), hide_index=True, width="stretch")

        q_chart_placeholder.line_chart(pd.DataFrame({"Q(s,a)": q_sample_history}), height=220)

        time.sleep(speed)
else:
    grid_placeholder.dataframe(render_grid(env.queues, env.lights), hide_index=True, width="stretch")
    metric_placeholder.info("Clique sur **Lancer la simulation** dans la barre latérale pour démarrer.")
