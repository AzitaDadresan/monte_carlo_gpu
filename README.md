# Monte Carlo GPU Option Pricing & Portfolio VaR

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![CuPy](https://img.shields.io/badge/CuPy-GPU-green.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

High-performance GPU Monte Carlo simulation for option pricing and portfolio risk (VaR/CVaR), inspired by AQR, Bridgewater, and CQF standards.

## Features
- **GPU-accelerated**: Uses CuPy for fast path simulation
- **Option pricing**: European call/put via Monte Carlo
- **Portfolio VaR/CVaR**: Risk metrics for multi-option portfolios
- **Configurable**: Supports custom option parameters and weights

## Requirements
- Python 3.11+
- cupy
- numpy

## Installation
```bash
pip install cupy numpy
```

## Usage
```bash
python monte_carlo_gpu.py
```

## Example
```python
from monte_carlo_gpu import Option, monte_carlo_gpu, portfolio_var_gpu

options = [
    Option(S0=490, K=500, T=0.25, r=0.02, sigma=0.42, type="call", weight=1.0),
    Option(S0=180, K=185, T=0.25, r=0.02, sigma=0.31, type="put",  weight=0.7),
    Option(S0=540, K=550, T=0.25, r=0.02, sigma=0.29, type="call", weight=1.2),
]
price = monte_carlo_gpu(options[0], n_paths=200_000)
var, cvar = portfolio_var_gpu(options, n_paths=50_000)
```

## License
MIT

## Author
Azita Dadresan | CQF, AQR, Bridgewater, JHU TA
