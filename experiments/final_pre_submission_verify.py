import pandas as pd
import numpy as np
import scipy.stats as stats

def run_audit():
    df = pd.read_csv('results/stage13/evaluation_episode_results.csv')
    c_df = df[df['method']=='cotop'].sort_values(by=['seed','episode']).reset_index(drop=True)
    l_df = df[df['method']=='local'].sort_values(by=['seed','episode']).reset_index(drop=True)
    g_df = df[df['method']=='greedy'].sort_values(by=['seed','episode']).reset_index(drop=True)

    # 1. CoTOP vs Local Delay (N=250)
    d_del = c_df['delay'].values - l_df['delay'].values
    N = len(d_del)
    df_freedom = N - 1
    mean_d_del = np.mean(d_del)
    std_d_del = np.std(d_del, ddof=1)
    sem_d_del = std_d_del / np.sqrt(N)
    t_del = mean_d_del / sem_d_del
    p_del = 2 * (1 - stats.t.cdf(abs(t_del), df=df_freedom))
    t_crit = stats.t.ppf(0.975, df=df_freedom)
    ci_del = [mean_d_del - t_crit * sem_d_del, mean_d_del + t_crit * sem_d_del]
    dz_del = mean_d_del / std_d_del
    cles_del = np.mean(c_df['delay'] < l_df['delay']) * 100.0

    print("=" * 70)
    print("1. CoTOP vs Local Latency (N = 250 Paired Episodes)")
    print("=" * 70)
    print(f"CoTOP Mean Delay:            {c_df['delay'].mean():.6f} s (std: {c_df['delay'].std(ddof=1):.6f})")
    print(f"Local Mean Delay:            {l_df['delay'].mean():.6f} s (std: {l_df['delay'].std(ddof=1):.6f})")
    print(f"Mean Difference (D_bar):     {mean_d_del:.6f} s (Reported: -0.0232 s)")
    print(f"Std of Differences (s_D):   {std_d_del:.6f} s (Reported: 0.3300 s)")
    print(f"Standard Error (SEM):        {sem_d_del:.6f} s (Reported: 0.0209 s)")
    print(f"Paired t-statistic:          {t_del:.6f} (Reported: -1.1121)")
    print(f"Degrees of Freedom:          {df_freedom}")
    print(f"p-value:                     {p_del:.6f} (Reported: 0.2672)")
    print(f"95% Confidence Interval:     [{ci_del[0]:.6f}, {ci_del[1]:.6f}] s (Reported: [-0.0643, +0.0179])")
    print(f"Cohen's dz:                  {dz_del:.6f} (Reported: -0.0703)")
    print(f"CLES (% CoTOP < Local):      {cles_del:.2f}% (Reported: 53.20%)")

    # 2. Hierarchical Seed-level (N=5)
    seeds = sorted(df['seed'].unique())
    seed_del_diffs = []
    for s in seeds:
        c_s = c_df[c_df['seed']==s]['delay'].values
        l_s = l_df[l_df['seed']==s]['delay'].values
        seed_del_diffs.append(np.mean(c_s - l_s))
    mean_s_del = np.mean(seed_del_diffs)
    std_s_del = np.std(seed_del_diffs, ddof=1)
    sem_s_del = std_s_del / np.sqrt(5)
    t_s_del = mean_s_del / sem_s_del
    p_s_del = 2 * (1 - stats.t.cdf(abs(t_s_del), df=4))
    t_crit_4 = stats.t.ppf(0.975, df=4)
    ci_s_del = [mean_s_del - t_crit_4 * sem_s_del, mean_s_del + t_crit_4 * sem_s_del]
    dz_s_del = mean_s_del / std_s_del

    print("\n" + "=" * 70)
    print("2. Hierarchical Seed-Level CoTOP vs Local (N = 5 Seeds)")
    print("=" * 70)
    print(f"Seed-Level Mean Difference:  {mean_s_del:.6f} s (Reported: -0.0232 s)")
    print(f"Seed-Level Std of Diff:      {std_s_del:.6f} s (Reported: 0.0647 s)")
    print(f"Seed-Level SEM:              {sem_s_del:.6f} s (Reported: 0.0289 s)")
    print(f"Seed-Level t-statistic:      {t_s_del:.6f} (Reported: -0.8018)")
    print(f"Degrees of Freedom:          4")
    print(f"p-value:                     {p_s_del:.6f} (Reported: 0.4676)")
    print(f"95% CI (df=4, t_crit=2.776): [{ci_s_del[0]:.6f}, {ci_s_del[1]:.6f}] s (Reported: [-0.1036, +0.0572])")
    print(f"Cohen's dz:                  {dz_s_del:.6f} (Reported: -0.3586)")

    # 3. CoTOP vs Greedy Energy (N=250)
    d_ene = c_df['energy'].values - g_df['energy'].values
    mean_c_ene = c_df['energy'].mean()
    mean_g_ene = g_df['energy'].mean()
    mean_d_ene = np.mean(d_ene)
    std_d_ene = np.std(d_ene, ddof=1)
    sem_d_ene = std_d_ene / np.sqrt(N)
    t_ene = mean_d_ene / sem_d_ene
    p_ene = 2 * (1 - stats.t.cdf(abs(t_ene), df=df_freedom))
    ci_ene = [mean_d_ene - t_crit * sem_d_ene, mean_d_ene + t_crit * sem_d_ene]
    dz_ene = mean_d_ene / std_d_ene
    cles_ene = np.mean(c_df['energy'] < g_df['energy']) * 100.0
    pct_ene = (mean_d_ene / mean_g_ene) * 100.0

    # Multiple testing adjustments
    p_vals = [p_ene, p_del, 0.3421, 0.5176]
    sorted_p = sorted(p_vals)
    holm_adj_p = min(1.0, sorted_p[0] * 4)
    bh_adj_p = min(1.0, sorted_p[0] * 4 / 1)

    print("\n" + "=" * 70)
    print("3. CoTOP vs Greedy Energy (N = 250 Paired Episodes)")
    print("=" * 70)
    print(f"CoTOP Mean Energy:           {mean_c_ene:.6f} J (Reported: 0.319 J)")
    print(f"Greedy Mean Energy:          {mean_g_ene:.6f} J (Reported: 4.525 J)")
    print(f"Mean Difference (D_bar):     {mean_d_ene:.6f} J (Reported: -4.2060 J)")
    print(f"Std of Differences (s_D):   {std_d_ene:.6f} J (Reported: 0.2764 J)")
    print(f"Standard Error (SEM):        {sem_d_ene:.6f} J (Reported: 0.0175 J)")
    print(f"Paired t-statistic:          {t_ene:.6f} (Reported: -240.58)")
    print(f"Degrees of Freedom:          {df_freedom}")
    print(f"p-value:                     {p_ene:.6e} (Reported: < 1e-140)")
    print(f"Holm-Bonferroni Adjusted p:  {holm_adj_p:.6e} (Reported: < 1e-4)")
    print(f"Benjamini-Hochberg Adjusted: {bh_adj_p:.6e} (Reported: < 1e-4)")
    print(f"95% Confidence Interval:     [{ci_ene[0]:.6f}, {ci_ene[1]:.6f}] J (Reported: [-4.2405, -4.1716])")
    print(f"Cohen's dz:                  {dz_ene:.6f} (Reported: -15.22)")
    print(f"CLES (% CoTOP < Greedy):     {cles_ene:.2f}% (Reported: 100.00%)")
    print(f"Percentage Energy Reduction: {pct_ene:.2f}% (Reported: -92.95%)")
    print("=" * 70)

if __name__ == '__main__':
    run_audit()
