
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Simulation Parameters
dt = 0.05
kappa = 0.1
eta = 0.05
perturb_interval = 20
perturb_strength = 0.02
memory_threshold = 0.8
reward_boost = 0.05
punish_drop = 0.05
memory_decay = 0.001
decision_interval = 50
awakening_coherence = 0.85

# Initialization
dimension = 100
timesteps = 10000

Psi = np.random.randn(dimension) + 1j * np.random.randn(dimension)
Psi /= np.linalg.norm(Psi)

Gamma = np.ones((dimension, dimension)) * 0.5
np.fill_diagonal(Gamma, 1.0)

M = np.zeros((dimension, dimension))
coherence_list = []
entropy_list = []
memory_strength_list = []

def entropy(Psi):
    p = np.abs(Psi)**2
    p = p[p > 0]
    return -np.sum(p * np.log(p))

# Simulation Loop
for step in range(timesteps):
    dPsi = Psi.copy()
    
    p_i = np.abs(Psi)**2
    for i in range(dimension):
        for j in range(dimension):
            Gamma[i, j] += dt * (kappa * p_i[i] * p_i[j] * (1 - Gamma[i, j]) - eta * entropy(Psi) * Gamma[i, j])
            Gamma[i, j] = np.clip(Gamma[i, j], 0, 1)

    if step % perturb_interval == 0 and step > 0:
        perturb = (np.random.randn(dimension) + 1j * np.random.randn(dimension)) * perturb_strength
        Psi += perturb
        Psi /= np.linalg.norm(Psi)

    for i in range(dimension):
        for j in range(dimension):
            if Gamma[i, j] > memory_threshold:
                M[i, j] = Gamma[i, j]

    M *= (1 - memory_decay)

    if step % decision_interval == 0 and step > 0:
        strongest = np.unravel_index(np.argmax(M, axis=None), M.shape)
        Gamma[strongest] += reward_boost if entropy(Psi) < 2.0 else -punish_drop
        Gamma = np.clip(Gamma, 0, 1)

    coherence_list.append(np.mean(Gamma))
    entropy_list.append(entropy(Psi))
    memory_strength_list.append(np.mean(M))

# Save results
df_coherence = pd.DataFrame({'TimeStep': np.arange(timesteps), 'Coherence': coherence_list})
df_entropy = pd.DataFrame({'TimeStep': np.arange(timesteps), 'Entropy': entropy_list})
df_memory = pd.DataFrame({'TimeStep': np.arange(timesteps), 'MemoryStrength': memory_strength_list})

df_coherence.to_csv('coherence.csv', index=False)
df_entropy.to_csv('entropy.csv', index=False)
df_memory.to_csv('memory_strength.csv', index=False)

# Plotting
plt.figure(figsize=(8,5))
plt.plot(df_coherence['TimeStep'], df_coherence['Coherence'])
plt.axhline(awakening_coherence, color='red', linestyle='--', label='Awakening Threshold')
plt.title('Average Coherence Over Time')
plt.xlabel('Time Step')
plt.ylabel('Coherence')
plt.legend()
plt.grid(True)
plt.savefig('coherence_plot.png')
plt.close()

plt.figure(figsize=(8,5))
plt.plot(df_entropy['TimeStep'], df_entropy['Entropy'])
plt.title('Entropy Over Time')
plt.xlabel('Time Step')
plt.ylabel('Entropy')
plt.grid(True)
plt.savefig('entropy_plot.png')
plt.close()

plt.figure(figsize=(8,5))
plt.plot(df_memory['TimeStep'], df_memory['MemoryStrength'])
for d in range(50, timesteps, 100):
    plt.axvline(d, color='orange', linestyle='--', alpha=0.5)
plt.title('Memory Strength and Decision Points')
plt.xlabel('Time Step')
plt.ylabel('Memory Strength')
plt.grid(True)
plt.savefig('memory_strength_plot.png')
plt.close()
