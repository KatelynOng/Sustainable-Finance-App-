import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Inputs from the user
# ------------------------------
r_h = float(input("Asset 1 Expected Return (%) [e.g., 5]: ")) / 100
sd_h = float(input("Asset 1 Standard Deviation (%) [e.g., 9]: ")) / 100
esg_h = float(input("Asset 1 ESG Score [0-7]: "))

r_f = float(input("Asset 2 Expected Return (%) [e.g., 12]: ")) / 100
sd_f = float(input("Asset 2 Standard Deviation (%) [e.g., 20]: ")) / 100
esg_f = float(input("Asset 2 ESG Score [0-7]: "))

rho_hf = float(input("Correlation between Asset 1 and 2 [-1 to 1, e.g., -0.2]: "))
r_free = float(input("Risk-Free Rate (%) [e.g., 2]: ")) / 100
gamma = float(input("Risk Aversion (γ) [e.g., 5]: "))
lambda_esg = float(input("ESG Preference (λ) [e.g., 0.05]: "))

# ------------------------------
# Functions
# ------------------------------
def portfolio_ret(w1, r1, r2):
    return w1 * r1 + (1 - w1) * r2

def portfolio_sd(w1, sd1, sd2, rho):
    return np.sqrt(w1**2 * sd1**2 + (1-w1)**2 * sd2**2 + 2 * rho * w1 * (1-w1) * sd1 * sd2)

def portfolio_esg(w1, esg1, esg2):
    return w1 * esg1 + (1 - w1) * esg2

def utility(ret, sd, esg, gamma, lambda_esg):
    return ret - 0.5 * gamma * sd**2 + lambda_esg * esg

# ------------------------------
# Tangency Portfolio
# ------------------------------
weights = np.linspace(0, 1, 1000)
sharpe_ratios = []

for w in weights:
    ret = portfolio_ret(w, r_h, r_f)
    sd = portfolio_sd(w, sd_h, sd_f, rho_hf)
    if sd > 0:
        sharpe = (ret - r_free) / sd
        sharpe_ratios.append(sharpe)
    else:
        sharpe_ratios.append(-np.inf)

max_idx = np.argmax(sharpe_ratios)
w1_tangency = weights[max_idx]
w2_tangency = 1 - w1_tangency

ret_tangency = portfolio_ret(w1_tangency, r_h, r_f)
sd_tangency = portfolio_sd(w1_tangency, sd_h, sd_f, rho_hf)

# ------------------------------
# Optimal Portfolio
# ------------------------------
utilities = []
returns = []
risk = []
esg_scores = []

if sd_tangency > 0:
    w_tangency_optimal = (ret_tangency - r_free) / (gamma * sd_tangency**2)
else:
    w_tangency_optimal = 0

for w in weights:
    esg = portfolio_esg(w, esg_h, esg_f)
    u = utility(ret, sd, esg, gamma, lambda_esg)

    utilities.append(u)
    returns.append(ret)
    risk.append(sd)
    esg_scores.append(esg)

w1_optimal = w_tangency_optimal * w1_tangency
w2_optimal = w_tangency_optimal * w2_tangency
w_rf_optimal = 1 - w_tangency_optimal

ret_optimal = r_free + w_tangency_optimal * (ret_tangency - r_free)
sd_optimal = abs(w_tangency_optimal) * sd_tangency
esg_optimal = esg_scores[max_idx]

max_idx = np.argmax(utilities)

w1_opt = weights[max_idx]
w2_opt = 1 - w1_opt

ret_opt = returns[max_idx]
sd_opt = risk[max_idx]
esg_opt = esg_scores[max_idx]

# ------------------------------
# Display results
# ------------------------------
print("\nOptimal Portfolio Weights:")
print(f"Risk-Free Asset: {w_rf_optimal*100:.2f}%")
print(f"Asset 1: {w1_optimal*100:.2f}%")
print(f"Asset 2: {w2_optimal*100:.2f}%")
print(f"Expected Return: {ret_optimal*100:.2f}%")
print(f"Portfolio Risk (Std Dev): {sd_optimal*100:.2f}%")
print(f"ESG Score: {esg_optimal:.2f}")

# ------------------------------
# Plot Efficient Frontier
# ------------------------------
weights_plot = np.linspace(0, 1, 200)
returns_frontier = [portfolio_ret(w, r_h, r_f) for w in weights_plot]
sds_frontier = [portfolio_sd(w, sd_h, sd_f, rho_hf) for w in weights_plot]
esg_frontier = [portfolio_esg(w, esg_h, esg_f) for w in weights_plot]

fig, ax = plt.subplots(figsize=(8, 5))

# Efficient frontier
ax.plot(sds_frontier, returns_frontier, 'b-', linewidth=2, label='Efficient Frontier')

# Capital Market Line
if sd_tangency > 0:
    sd_max = max(sds_frontier) * 1.2
    sd_cml = np.linspace(0, sd_max, 100)
    ret_cml = r_free + (ret_tangency - r_free) / sd_tangency * sd_cml
    ax.plot(sd_cml, ret_cml, 'g--', linewidth=2, label='Capital Market Line')

# Tangency portfolio
ax.scatter(sd_tangency, ret_tangency, color='red', s=100, marker='*', label='Tangency Portfolio')

# Optimal portfolio
ax.scatter(sd_optimal, ret_optimal, color='orange', s=100, marker='D', label='Optimal Portfolio')

# Risk-free asset
ax.scatter(0, r_free, color='green', s=80, marker='s', label='Risk-Free Asset')

# Scatter with ESG coloring
sc = ax.scatter(sds_frontier, returns_frontier, c=esg_frontier, cmap='viridis', label='Portfolio Set')

# Color bar
cbar = plt.colorbar(sc)
cbar.set_label('ESG Score')

# Optimal portfolio point
ax.scatter(sd_opt, ret_opt, color='red', s=120, marker='*', label='Optimal Portfolio')

# Labels
ax.set_xlabel('Risk (Standard Deviation)')
ax.set_ylabel('Expected Return')
ax.set_title('ESG-Efficient Frontier')
ax.legend()
ax.grid(True, alpha=0.3)

plt.show()