import numpy as np
from numba import njit
from utils.environments import sample_reward

@njit
def generate_harmonic_schedule(k, max_blocks):
    """Pre-computes the rank exploration schedule iteratively appending divisors."""
    schedule = np.empty(max_blocks, dtype=np.int64)
    idx = 0
    n = 1
    
    while idx < max_blocks:
        for d in range(1, min(n, k) + 1):
            if n % d == 0:
                schedule[idx] = d
                idx += 1
                if idx >= max_blocks:
                    break
        n += 1
        
    return schedule

@njit
def simulate_hare(means, T, sigma2, env_type, seed=0):
    """
    Runs a single trial of the UCB-HARE algorithm.
    Uses the divisor harmonic schedule and the paper's block-boundary stopping rule.
    The confidence level is delta=1/T, as used for the paper's main guarantee.
    Returns chosen arms and the run-level expected mean $m_tilde at each round.
    """
    np.random.seed(seed)
    k = len(means)
    
    n_pulls = np.zeros(k, dtype=np.int64)
    sums = np.zeros(k, dtype=np.float64)
    mu_hat = np.zeros(k, dtype=np.float64)
    arms = np.empty(T, dtype=np.int64)
    m_tilde = np.empty(T, dtype=np.float64)
    
    # Paper's delta=1/T setting: L = log(8*k*T/delta).
    L = np.log(8.0 * k * T * T)
    
    pi = np.arange(k)
    np.random.shuffle(pi)
    
    # Phase I cannot use more than ceil(T/2) complete two-round blocks.
    # Allocate the schedule to that exact horizon so a long transient never
    # exhausts an implementation-only schedule cap.
    schedule = generate_harmonic_schedule(k, (T + 1) // 2)
    
    t = 0
    b = 1
    phase_1_active = True
    
    # ---------------- Phase I: Preparation ----------------
    while phase_1_active and t < T:
        
        # The block decides the interleaving pattern in advance
        # The paper labels scheduled-first as theta_b=1. Invert the raw bit to
        # retain that convention and the existing seeded simulation sequence.
        theta_b = 1 - np.random.randint(2)
        scheduled_first = theta_b == 1
        
        for slot in range(2):
            if t >= T: 
                phase_1_active = False
                break
                
            # --- CALCULATE ANCHOR AT EVERY SINGLE TIME STEP ---
            B_t = -1e9
            best_anchor = 0
            for i in range(k):
                if n_pulls[i] > 0:
                    c_n = np.sqrt((2.0 * sigma2 * L) / float(n_pulls[i]))
                    lower_bound = mu_hat[i] - c_n
                    if lower_bound > B_t:
                        B_t = lower_bound
                        best_anchor = i
                else:
                    c_n = 1e9 
                    
            min_n = np.min(n_pulls)
            
            # Keep stopping checks at block boundaries, as in the analysis.
            if slot == 0 and B_t > 0 and min_n >= (32.0 * sigma2 * L) / (B_t**2):
                phase_1_active = False
                break
                
            is_scheduled_slot = ((slot == 0) == scheduled_first)
            
            if is_scheduled_slot or B_t <= 0:
                # Scheduled slot OR anchorless fallback
                rank_to_pull = schedule[b - 1] - 1 
                chosen_arm = pi[rank_to_pull]
            else:
                # Auxiliary slot with a perfectly fresh, valid anchor
                chosen_arm = best_anchor

            scheduled_arm = pi[schedule[b - 1] - 1]
            auxiliary_arm = best_anchor if B_t > 0 else scheduled_arm
            if slot == 0:
                first_choice_scheduled_first = scheduled_arm
                first_choice_auxiliary_first = auxiliary_arm
                m_tilde[t] = 0.5 * (means[scheduled_arm] + means[auxiliary_arm])
            else:
                w_scheduled_first = 1.0 if arms[t - 1] == first_choice_scheduled_first else 0.0
                w_auxiliary_first = 1.0 if arms[t - 1] == first_choice_auxiliary_first else 0.0
                if w_scheduled_first + w_auxiliary_first == 0.0:
                    w_scheduled_first = 0.5
                    w_auxiliary_first = 0.5
                auxiliary_now = best_anchor if B_t > 0 else scheduled_arm
                m_tilde[t] = (
                    w_scheduled_first * means[auxiliary_now]
                    + w_auxiliary_first * means[scheduled_arm]
                ) / (w_scheduled_first + w_auxiliary_first)
                
            # Observe and update
            r = sample_reward(means[chosen_arm], sigma2, env_type)
            n_pulls[chosen_arm] += 1
            sums[chosen_arm] += r
            mu_hat[chosen_arm] = sums[chosen_arm] / n_pulls[chosen_arm]
            arms[t] = chosen_arm
            t += 1
            
        b += 1

    # ---------------- Phase II: UCB Exploitation ----------------
    while t < T:
        best_ucb = -1e9
        chosen_arm = 0
        
        for i in range(k):
            if n_pulls[i] > 0:
                bonus = np.sqrt((2.0 * sigma2 * L) / float(n_pulls[i]))
                ucb_val = mu_hat[i] + bonus
            else:
                ucb_val = 1e9
                
            if ucb_val > best_ucb:
                best_ucb = ucb_val
                chosen_arm = i
                
        m_tilde[t] = means[chosen_arm]
        r = sample_reward(means[chosen_arm], sigma2, env_type)
        n_pulls[chosen_arm] += 1
        sums[chosen_arm] += r
        mu_hat[chosen_arm] = sums[chosen_arm] / n_pulls[chosen_arm]
        arms[t] = chosen_arm
        t += 1
        
    return arms, m_tilde
