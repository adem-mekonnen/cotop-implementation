import pandas as pd
import numpy as np
from scipy import stats

def compute_exact_ground_truth():
    df = pd.read_csv("results/stage13/evaluation_episode_results.csv")
    
    # Filter methods
    df_cotop = df[df['method'] == 'cotop'].sort_values(by=['seed', 'episode']).reset_index(drop=True)
    df_local = df[df['method'] == 'local'].sort_values(by=['seed', 'episode']).reset_index(drop=True)
    df_greedy = df[df['method'] == 'greedy'].sort_values(by=['seed', 'episode']).reset_index(drop=True)
    
    N = len(df_cotop)
    df_freedom = N - 1
    
    # -------------------------------------------------------------
    # 1. CoTOP vs Local (Delay)
    # -------------------------------------------------------------
    d_del_loc = df_cotop['delay'].values - df_local['delay'].values
    mean_d_del_loc = np.mean(d_del_loc)
    std_d_del_loc = np.std(d_del_loc, ddof=1)
    sem_d_del_loc = std_d_del_loc / np.sqrt(N)
    t_del_loc = mean_d_del_loc / sem_d_del_loc
    p_del_loc = 2 * (1 - stats.t.cdf(abs(t_del_loc), df=df_freedom))
    t_crit = stats.t.ppf(0.975, df=df_freedom)
    ci_del_loc = (mean_d_del_loc - t_crit * sem_d_del_loc, mean_d_del_loc + t_crit * sem_d_del_loc)
    dz_del_loc = mean_d_del_loc / std_d_del_loc
    w_del_loc = stats.wilcoxon(d_del_loc)
    cles_del_loc = np.mean(df_cotop['delay'] < df_local['delay']) * 100.0
    
    print("=" * 60)
    print("EXACT GROUND TRUTH: CoTOP vs Local (Total Delay)")
    print("=" * 60)
    print(f"N: {N}, df: {df_freedom}")
    print(f"CoTOP Mean Delay: {df_cotop['delay'].mean():.6f} s (std: {df_cotop['delay'].std(ddof=1):.6f})")
    print(f"Local Mean Delay: {df_local['delay'].mean():.6f} s (std: {df_local['delay'].std(ddof=1):.6f})")
    print(f"Mean Difference (D_bar): {mean_d_del_loc:.6f} s")
    print(f"Std of Differences (s_D): {std_d_del_loc:.6f} s")
    print(f"Standard Error (SEM): {sem_d_del_loc:.6f} s")
    print(f"t-statistic: {t_del_loc:.6f}")
    print(f"p-value: {p_del_loc:.6f}")
    print(f"Wilcoxon W: {w_del_loc.statistic}, p-value: {w_del_loc.pvalue:.6f}")
    print(f"95% CI: [{ci_del_loc[0]:.6f}, {ci_del_loc[1]:.6f}] s")
    print(f"Cohen's dz: {dz_del_loc:.6f}")
    print(f"CLES (% CoTOP < Local): {cles_del_loc:.2f}%")
    
    # -------------------------------------------------------------
    # 2. CoTOP vs Greedy (Energy)
    # -------------------------------------------------------------
    d_ene_gr = df_cotop['energy'].values - df_greedy['energy'].values
    mean_d_ene_gr = np.mean(d_ene_gr)
    std_d_ene_gr = np.std(d_ene_gr, ddof=1)
    sem_d_ene_gr = std_d_ene_gr / np.sqrt(N)
    t_ene_gr = mean_d_ene_gr / sem_d_ene_gr
    p_ene_gr = 2 * (1 - stats.t.cdf(abs(t_ene_gr), df=df_freedom))
    ci_ene_gr = (mean_d_ene_gr - t_crit * sem_d_ene_gr, mean_d_ene_gr + t_crit * sem_d_ene_gr)
    dz_ene_gr = mean_d_ene_gr / std_d_ene_gr
    w_ene_gr = stats.wilcoxon(d_ene_gr)
    pct_ene_gr = (mean_d_ene_gr / df_greedy['energy'].mean()) * 100.0
    cles_ene_gr = np.mean(df_cotop['energy'] < df_greedy['energy']) * 100.0
    
    print("\n" + "=" * 60)
    print("EXACT GROUND TRUTH: CoTOP vs Greedy (Total Energy)")
    print("=" * 60)
    print(f"CoTOP Mean Energy: {df_cotop['energy'].mean():.6f} J (std: {df_cotop['energy'].std(ddof=1):.6f})")
    print(f"Greedy Mean Energy: {df_greedy['energy'].mean():.6f} J (std: {df_greedy['energy'].std(ddof=1):.6f})")
    print(f"Mean Difference (D_bar): {mean_d_ene_gr:.6f} J")
    print(f"Percentage Reduction: {pct_ene_gr:.2f}%")
    print(f"Std of Differences (s_D): {std_d_ene_gr:.6f} J")
    print(f"Standard Error (SEM): {sem_d_ene_gr:.6f} J")
    print(f"t-statistic: {t_ene_gr:.6f}")
    print(f"p-value: {p_ene_gr:.6e}")
    print(f"Wilcoxon W: {w_ene_gr.statistic}, p-value: {w_ene_gr.pvalue:.6e}")
    print(f"95% CI: [{ci_ene_gr[0]:.6f}, {ci_ene_gr[1]:.6f}] J")
    print(f"Cohen's dz: {dz_ene_gr:.6f}")
    print(f"CLES (% CoTOP < Greedy): {cles_ene_gr:.2f}%")
    
    # -------------------------------------------------------------
    # 3. Seed-level analysis (N=5 seeds)
    # -------------------------------------------------------------
    seeds = [42, 123, 456, 789, 2026]
    seed_stats = []
    for s in seeds:
        c_del = df_cotop[df_cotop['seed'] == s]['delay'].mean()
        l_del = df_local[df_local['seed'] == s]['delay'].mean()
        g_ene = df_greedy[df_greedy['seed'] == s]['energy'].mean()
        c_ene = df_cotop[df_cotop['seed'] == s]['energy'].mean()
        seed_stats.append({
            'seed': s,
            'cotop_delay': c_del,
            'local_delay': l_del,
            'diff_delay': c_del - l_del,
            'cotop_energy': c_ene,
            'greedy_energy': g_ene,
            'diff_energy': c_ene - g_ene
        })
    df_seed = pd.DataFrame(seed_stats)
    print("\n" + "=" * 60)
    print("SEED-LEVEL SUMMARY (N=5 independent seeds)")
    print("=" * 60)
    print(df_seed.to_string(index=False))
    
    # Seed-level t-test for delay diff (df=4)
    seed_del_diffs = df_seed['diff_delay'].values
    seed_t_del = np.mean(seed_del_diffs) / (np.std(seed_del_diffs, ddof=1) / np.sqrt(5))
    seed_p_del = 2 * (1 - stats.t.cdf(abs(seed_t_del), df=4))
    t_crit_4 = stats.t.ppf(0.975, df=4)
    seed_ci_del = (np.mean(seed_del_diffs) - t_crit_4 * (np.std(seed_del_diffs, ddof=1) / np.sqrt(5)),
                   np.mean(seed_del_diffs) + t_crit_4 * (np.std(seed_del_diffs, ddof=1) / np.sqrt(5)))
    
    print(f"\nSeed-level Delay Diff Mean: {np.mean(seed_del_diffs):.6f} s")
    print(f"Seed-level Delay Diff Std: {np.std(seed_del_diffs, ddof=1):.6f} s")
    print(f"Seed-level paired t(4): {seed_t_del:.4f}, p-value: {seed_p_del:.4f}")
    print(f"Seed-level 95% CI (df=4): [{seed_ci_del[0]:.6f}, {seed_ci_del[1]:.6f}] s")

if __name__ == "__main__":
    compute_exact_ground_truth()
