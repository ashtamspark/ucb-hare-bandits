import numpy as np
from numba import njit
from utils.environments import sample_reward

@njit
def phase1_condition(mu, n, sigma2, logT, c, p):
    k = len(mu)
    for i in range(k):
        if n[i] == 0:  
            continue

        bonus = c * np.sqrt((2 * sigma2 * logT) / n[i])
        lhs1 = mu[i] <= bonus

        denom = mu[i] - bonus
        if denom <= 0:
            rhs_term = 1e18  
        else:
            rhs_term = (200 * (c**2) * (p**2) * sigma2 * logT) / denom \
                       + c * np.sqrt(2 * n[i] * sigma2 * logT)

        lhs2 = (n[i] * mu[i]) < rhs_term

        if not (lhs1 or lhs2):
            return False
    return True

@njit
def simulate_welfarist(means, T, sigma2, env_type, p, seed=0):
    """Runs a single trial of the Welfarist UCB algorithm."""
    np.random.seed(seed)
    k = len(means)
    n = np.zeros(k, dtype=np.int64)
    sums = np.zeros(k, dtype=np.float64)
    mu = np.zeros(k, dtype=np.float64)
    arms = np.empty(T, dtype=np.int64)
    
    c = 2.0
    t = 0
    logT = np.log(float(T))
    
    B = np.arange(k)
    np.random.shuffle(B)
    b_index = 0

    p_a = 1.0 if p >= -1.0 else float(p)

    # ---- Phase 1 ----
    while phase1_condition(mu, n, sigma2, logT, c, p_a):
        if t % k == 0:
            B = np.arange(k)
            np.random.shuffle(B)
            b_index = 0

        a = B[b_index]
        b_index += 1

        r = sample_reward(means[a], sigma2, env_type)
        n[a] += 1
        sums[a] += r
        mu[a] = sums[a] / n[a]
        arms[t] = a
        t += 1
        if t >= T:
            break

    # ---- Phase 2 ----
    while t < T:
        best_ucb = -1e9
        chosen_arm = 0
        for a in range(k):
            if n[a] > 0:
                bonus = c * np.sqrt((2 * sigma2 * logT) / n[a])
                ucb_val = mu[a] + bonus
            else:
                ucb_val = 1e9  
                
            if ucb_val > best_ucb:
                best_ucb = ucb_val
                chosen_arm = a
                
        r = sample_reward(means[chosen_arm], sigma2, env_type)
        n[chosen_arm] += 1
        sums[chosen_arm] += r
        mu[chosen_arm] = sums[chosen_arm] / n[chosen_arm]
        arms[t] = chosen_arm
        t += 1
    
    return arms