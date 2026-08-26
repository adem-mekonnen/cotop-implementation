import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('results/stage13/evaluation_episode_results.csv')
df_c = df[df['method']=='cotop']
df_l = df[df['method']=='local']
df_g = df[df['method']=='greedy']

seeds = sorted(df['seed'].unique())
print("Available seeds:", seeds)
seed_del_diffs = []
seed_ene_diffs = []

for s in seeds:
    c_d = df_c[df_c['seed']==s]['delay'].values
    l_d = df_l[df_l['seed']==s]['delay'].values
    c_e = df_c[df_c['seed']==s]['energy'].values
    g_e = df_g[df_g['seed']==s]['energy'].values
    
    del_diff = np.mean(c_d - l_d)
    ene_diff = np.mean(c_e - g_e)
    seed_del_diffs.append(del_diff)
    seed_ene_diffs.append(ene_diff)
    
    print(f"Seed {s}: CoTOP delay={np.mean(c_d):.4f}s, Local delay={np.mean(l_d):.4f}s, diff={del_diff:.4f}s | CoTOP energy={np.mean(c_e):.4f}J, Greedy energy={np.mean(g_e):.4f}J, diff={ene_diff:.4f}J")

mean_seed_del_diff = np.mean(seed_del_diffs)
std_seed_del_diff = np.std(seed_del_diffs, ddof=1)
sem_seed_del_diff = std_seed_del_diff / np.sqrt(len(seeds))
t_seed_del = mean_seed_del_diff / sem_seed_del_diff
p_seed_del = 2 * (1 - stats.t.cdf(abs(t_seed_del), df=len(seeds)-1))

print("\n--- Seed-level (N=5) Delay Difference Summary ---")
print(f"Mean: {mean_seed_del_diff:.6f} s, Std: {std_seed_del_diff:.6f} s, SEM: {sem_seed_del_diff:.6f} s")
print(f"t(4) = {t_seed_del:.4f}, p-value = {p_seed_del:.4f}")
t_crit_4 = stats.t.ppf(0.975, df=4)
ci_seed = [mean_seed_del_diff - t_crit_4 * sem_seed_del_diff, mean_seed_del_diff + t_crit_4 * sem_seed_del_diff]
print(f"Seed-level 95% CI (df=4): [{ci_seed[0]:.6f}, {ci_seed[1]:.6f}] s")
