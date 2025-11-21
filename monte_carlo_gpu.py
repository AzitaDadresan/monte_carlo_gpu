import cupy as cp
import numpy as np
from dataclasses import dataclass

# -------------------------------
# Config
# -------------------------------
@dataclass
class Option:
    S0: float     # initial price
    K: float      # strike
    T: float      # maturity (in years)
    r: float      # risk-free rate
    sigma: float  # volatility
    type: str     # "call" or "put"
    weight: float # portfolio weight

# -----------------------------------------------------
# GPU Monte Carlo Kernel 
# -----------------------------------------------------
def monte_carlo_gpu(option: Option, n_paths=100_000, n_steps=252):
    dt = option.T / n_steps
    
    # Generate all random shocks on GPU
    # Ensure stable variance at high path counts
    Z = cp.random.standard_normal(size=(n_paths, n_steps), dtype=cp.float32)
    
    # Pre-allocate paths
    S = cp.full((n_paths,), option.S0, dtype=cp.float32)
    
    drift = (option.r - 0.5 * option.sigma ** 2) * dt
    vol = option.sigma * cp.sqrt(dt)

    # GPU time-stepping loop
    for t in range(n_steps):
        S *= cp.exp(drift + vol * Z[:, t])
    
    # Payoff (branchless for speed)
    if option.type.lower() == "call":
        payoff = cp.maximum(S - option.K, 0.0)
    else:
        payoff = cp.maximum(option.K - S, 0.0)
    
    price = cp.mean(payoff) * cp.exp(-option.r * option.T)

    return float(price)

# ---------------------------------------------------------------------
# Portfolio VaR / CVaR Calculator 
# ---------------------------------------------------------------------
def portfolio_var_gpu(options, n_paths=100_000, n_steps=252, alpha=0.99):
    # Compute terminal portfolio value for each path
    pnl_matrix = []
    for opt in options:
        price_paths = monte_carlo_gpu(opt, n_paths=n_paths, n_steps=n_steps)
        pnl_matrix.append(price_paths * opt.weight)
    portfolio_end = np.sum(pnl_matrix, axis=0) if isinstance(pnl_matrix[0], np.ndarray) else pnl_matrix
    portfolio_end = np.array(portfolio_end)
    # Convert to P&L vector
    pnl = portfolio_end - portfolio_end.mean()
    # VaR
    var = np.percentile(pnl, (1 - alpha) * 100)
    # CVaR (expected shortfall)
    cvar = pnl[pnl <= var].mean()
    return var, cvar

# -----------------------------------------------------------------
# Example Usage 
# -----------------------------------------------------------------
if __name__ == "__main__":
    options = [
        Option(S0=490, K=500, T=0.25, r=0.02, sigma=0.42, type="call", weight=1.0),  # NVDA
        Option(S0=180, K=185, T=0.25, r=0.02, sigma=0.31, type="put",  weight=0.7),  # AAPL
        Option(S0=540, K=550, T=0.25, r=0.02, sigma=0.29, type="call", weight=1.2),  # SPY
    ]
    print("Running GPU Monte Carlo...")
    price = monte_carlo_gpu(options[0], n_paths=200_000)
    print("Example Option Price:", price)
    print("Computing portfolio VaR...")
    var, cvar = portfolio_var_gpu(options, n_paths=50_000)
    print("VaR(99%):", var)
    print("CVaR(99%):", cvar)
