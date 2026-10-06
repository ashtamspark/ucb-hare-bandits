import numpy as np
from numba import njit
from utils.environments import sample_reward

@njit
def simulate_explore_ucb(means, T, sigma2, env_type, p, seed=0):
    """Runs a single trial of the Explore-Then-UCB baseline."""
    np.random.seed(seed)
    k = len(means)
    n = np.zeros(k, dtype=np.int64)
    sums = np.zeros(k, dtype=np.float64)
    mu_hat = np.zeros(k, dtype=np.float64)
    arms = np.empty(T, dtype=np.int64)
    
    den = np.log(k) if p >= 0 else 1.0
    if p == 0:
        tilde_T = int(16 * np.sqrt(T * k * np.log(T) / den))
    elif p > 0: 
        tilde_T = int(16 * np.sqrt(T * (float(k)**p) * np.log(T) / den))
    else:
        tilde_T = int(16 * np.sqrt(T * np.log(T) / (float(k)**-p)))
        
    t = 0
    
    # ---- Phase 1: Exploration ----
    while t < tilde_T and t < T:
        i_t = np.random.randint(k)
        r = sample_reward(means[i_t], sigma2, env_type)
        
        n[i_t] += 1
        sums[i_t] += r
        mu_hat[i_t] = sums[i_t] / n[i_t]
        arms[t] = i_t
        t += 1
    
    # ---- Phase 2: Exploitation with UCB ----
    while t < T:
        best_ucb = -1e9
        chosen_arm = 0
        
        for i in range(k):
            if n[i] > 0:
                bonus = 4 * np.sqrt(np.log(T) / n[i])
                ucb_val = mu_hat[i] + bonus
            else:
                ucb_val = 1e9
                
            if ucb_val > best_ucb:
                best_ucb = ucb_val
                chosen_arm = i
                
        r = sample_reward(means[chosen_arm], sigma2, env_type)
        
        n[chosen_arm] += 1
        sums[chosen_arm] += r
        mu_hat[chosen_arm] = sums[chosen_arm] / n[chosen_arm]
        arms[t] = chosen_arm
        t += 1
    
    return arms