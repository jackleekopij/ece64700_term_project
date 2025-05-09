import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

N = 5  
T = 24 
rho = 0.01


t_hours = np.arange(T)
p_base = 10 + 5 * np.sin(np.pi * (t_hours - 8) / 12) + 2 * np.sin(np.pi * (t_hours) / 6)
p_base[p_base < 5] = 5
p_base = p_base / 10

s_dts = 0.1 * np.sin(np.pi * (t_hours - 2) / 12)
p_effective = p_base - s_dts

np.random.seed(42)
x_min_val = -np.random.uniform(1, 3, N) 
x_max_val = np.random.uniform(3, 6, N)  
E_i_targets = np.random.uniform(-5, 10, N) 


B_min_soc = 5.0  
B_max_soc = 50.0  
P_min_batt = -20.0 
P_max_batt = 20.0  
eta_c = 0.95
eta_d = 0.95
b_initial_soc = (B_max_soc + B_min_soc) / 2 # Renamed


plt.figure(figsize=(10, 4))
plt.plot(t_hours, p_base, label='Base Price $p(t)$', color='blue')
plt.plot(t_hours, s_dts, label='DTS $s(t)$', color='green', linestyle='--')
plt.plot(t_hours, p_effective, label='Effective Price $p_{eff}(t)$', color='red', linestyle=':')
plt.xlabel('Time (hour)')
plt.ylabel('Price Signal (normalized)')
plt.title('Price and DTS Signals')
plt.legend()
plt.savefig("figure_prices.png", dpi=300)
plt.close()

lambda_values_to_test = [0, 0.1, 1, 5]
all_run_results = {}
colors = ['C0', 'C1', 'C2', 'C3']
linestyles = ['-', '-', '-', '-']


for idx, lambda_consensus in enumerate(lambda_values_to_test):
    # ADMM Initialization 
    x_profiles = np.zeros((N, T)) 
    z_aggregate = np.zeros(T)      
    b_soc_trajectory = np.zeros(T + 1)
    b_soc_trajectory[0] = b_initial_soc
    mu_dual = np.zeros(T) 
    primal_residuals = []
    dual_residuals = []
    MAX_ITER = 1000
    TOLERANCE = 1e-3
    solver_to_use = cp.OSQP 

    converged = False
    k_iter = 0
    for k_iter_loop in range(MAX_ITER):
        k_iter = k_iter_loop
        x_prev = x_profiles.copy()
        z_prev = z_aggregate.copy()

        # --- Step 1: Local User Updates ---
        x_bar_k = np.mean(x_profiles, axis=0)
        x_sum_k = np.sum(x_profiles, axis=0)
        for i_user in range(N):
            x_i_var = cp.Variable(T)
            v_i_k = z_prev - (x_sum_k - x_profiles[i_user, :])


            objective_i = cp.sum(
                cp.multiply(p_effective, x_i_var) +
                lambda_consensus * cp.square(x_i_var - x_bar_k) +
                cp.multiply(mu_dual, x_i_var) + 
                (rho / 2) * cp.square(x_i_var - v_i_k) )

            constraints_i = [x_i_var >= x_min_val[i_user], x_i_var <= x_max_val[i_user], cp.sum(x_i_var) == E_i_targets[i_user]]
            problem_i = cp.Problem(cp.Minimize(objective_i), constraints_i)

            problem_i.solve(solver=solver_to_use, verbose=False, max_iter=6000, warm_start=True)
            if problem_i.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and x_i_var.value is not None:
                x_profiles[i_user, :] = x_i_var.value
            else:
                x_profiles[i_user, :] = x_prev[i_user, :] 


        # Global Consensus Update 
        z_charge_var = cp.Variable(T, nonneg=True)
        z_discharge_var = cp.Variable(T, nonneg=True)
        z_net_var = z_charge_var - z_discharge_var
        b_var_step2 = cp.Variable(T + 1)
        x_sum_updated = np.sum(x_profiles, axis=0) 

        objective_z = cp.sum(cp.multiply(-mu_dual, z_net_var) + (rho / 2) * cp.square(x_sum_updated - z_net_var))
        constraints_z = [
            z_net_var >= P_min_batt, z_net_var <= P_max_batt, b_var_step2[0] == b_initial_soc,
            b_var_step2[1:] == b_var_step2[:-1] - (eta_c * z_charge_var - (1/eta_d) * z_discharge_var), 
            b_var_step2 >= B_min_soc, b_var_step2 <= B_max_soc
        ]
        problem_z = cp.Problem(cp.Minimize(objective_z), constraints_z)
        problem_z.solve(solver=solver_to_use, verbose=False, max_iter=6000, warm_start=True)
        if problem_z.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and z_net_var.value is not None and b_var_step2.value is not None :
            z_aggregate = z_net_var.value
            b_soc_trajectory = b_var_step2.value
        else: 
            z_aggregate = z_prev
            temp_b = np.zeros(T+1); temp_b[0]=b_initial_soc
            for t_idx in range(T): temp_b[t_idx+1] = np.clip(temp_b[t_idx] - eta_c*max(0,z_prev[t_idx]) + (1/eta_d)*max(0,-z_prev[t_idx]), B_min_soc, B_max_soc) # Assuming positive z is charge
            b_soc_trajectory = temp_b

        # Dual Variable Update 
        primal_res_val = np.sum(x_profiles, axis=0) - z_aggregate
        mu_dual = mu_dual + rho * primal_res_val

        # Check Convergence 
        primal_norm = np.linalg.norm(primal_res_val)
        dual_norm = np.linalg.norm(rho * (z_aggregate - z_prev))
        primal_residuals.append(primal_norm)
        dual_residuals.append(dual_norm)

        if np.isnan(primal_norm) or np.isnan(dual_norm):
            break
        if k_iter > 0 and primal_norm < TOLERANCE and dual_norm < TOLERANCE:
            converged = True
            break
        if (k_iter+1) % 50 == 0:
            print(f"λ={lambda_consensus}, It{k_iter+1}: PrRes={primal_norm:.3e}, DuRes={dual_norm:.3e}")

    iterations_taken = k_iter + 1
    if not converged:
        print(f"ADMM did NOT converge within {MAX_ITER} iter. Final Res: Pr={primal_norm:.3e}, Du={dual_norm:.3e}")

    # Create data structure for all results
    all_run_results[lambda_consensus] = {
        'x_profiles': x_profiles.copy(), 'z_aggregate': z_aggregate.copy(),
        'b_soc_trajectory': b_soc_trajectory.copy(),
        'primal_residuals': primal_residuals.copy(), 'dual_residuals': dual_residuals.copy(),
        'user_profile_variance_t': user_profile_variance_t.copy(),
        'converged': converged, 'iterations_taken': iterations_taken,
        'total_cost_effective': total_cost_effective,
        'actual_energy_cost_total': actual_energy_cost_total,
        'consensus_penalty_total': consensus_penalty_total
    }

# ADMM Convergence 
fig_conv, axes_conv = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
for idx, lambda_val in enumerate(lambda_values_to_test):
    if lambda_val in all_run_results:
        results = all_run_results[lambda_val]
        p_res = results['primal_residuals']
        d_res = results['dual_residuals']
        iters_taken = results['iterations_taken']
        label_text = f'λ={lambda_val} ({iters_taken} iters)'
        if len(p_res) > 0:
            axes_conv[0].semilogy(np.arange(1, len(p_res) + 1), p_res, label=label_text, color=colors[idx], linestyle=linestyles[idx])
        if len(d_res) > 0:
            axes_conv[1].semilogy(np.arange(1, len(d_res) + 1), d_res, label=label_text, color=colors[idx], linestyle=linestyles[idx])

axes_conv[0].set_xlabel('Iteration Count')
axes_conv[0].set_ylabel('Primal Residual Norm (log scale)')
axes_conv[0].set_title('ADMM Primal Residual Convergence')
axes_conv[0].axhline(TOLERANCE, color='k', linestyle='--', label='Tolerance', alpha=0.7)
axes_conv[0].legend(fontsize=9)
axes_conv[0].grid(True, which="both", ls="-", alpha=0.5)

axes_conv[1].set_xlabel('Iteration Count')
axes_conv[1].set_ylabel('Dual Residual Norm (log scale)')
axes_conv[1].set_title('ADMM Dual Residual Convergence')
axes_conv[1].axhline(TOLERANCE, color='k', linestyle='--', label='Tolerance', alpha=0.7)
axes_conv[1].legend(fontsize=9)


plt.savefig("figure_combined_admm_convergence.png", dpi=300)
plt.show()
plt.close()

# Aggregate Load z_aggregate(t) 
plt.figure(figsize=(12, 6))
for idx, lambda_val in enumerate(lambda_values_to_test):
    if lambda_val in all_run_results:
        results = all_run_results[lambda_val]
        if results['z_aggregate'].size == T: # Ensure z_aggregate has data
            plt.plot(t_hours, results['z_aggregate'], label=f'$\lambda={lambda_val}$', color=colors[idx], linestyle=linestyles[idx], linewidth=1.5)
plt.axhline(P_max_batt, color='grey', linestyle=':', linewidth=1, label=f'$P_{{max}}^{{batt}}$ ({P_max_batt} kW)')
plt.axhline(P_min_batt, color='grey', linestyle=':', linewidth=1, label=f'$P_{{min}}^{{batt}}$ ({P_min_batt} kW)')
plt.xlabel('Time (hour)')
plt.ylabel('Aggregate Load $z(t)$ (kW)')
plt.title('Aggregate Load Profiles (Battery Net Power) for Different $\lambda$ Values', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.7)
plt.axhline(0, color='black', linewidth=0.5, alpha=0.5)
plt.savefig("figure_combined_aggregate_load.png", dpi=300)
plt.close()

# Battery SOC b(t)
plt.figure(figsize=(12, 6))
for idx, lambda_val in enumerate(lambda_values_to_test):
    if lambda_val in all_run_results:
        results = all_run_results[lambda_val]
        if results['b_soc_trajectory'].size == T + 1:
            plt.plot(np.arange(T + 1), results['b_soc_trajectory'], label=f'$\lambda={lambda_val}$', color=colors[idx], linestyle=linestyles[idx], linewidth=1.5)
plt.axhline(B_max_soc, color='grey', linestyle=':', linewidth=1, label=f'$B_{{max}}$ ({B_max_soc} kWh)')
plt.axhline(B_min_soc, color='grey', linestyle=':', linewidth=1, label=f'$B_{{min}}$ ({B_min_soc} kWh)')
plt.xlabel('Time (hour)')
plt.ylabel('Battery SOC (kWh)')
plt.title('Battery SOC Trajectories for Different $\lambda$ Values', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.7)
plt.savefig("figure_combined_battery_soc.png", dpi=300)
plt.show()
plt.close()
