# Monte Carlo Option Pricer

A Monte Carlo simulation engine for pricing European vanilla options under Black-Scholes assumptions.

## Features

- **European Call/Put Options**: Price vanilla European options
- **Risk-Neutral GBM**: Uses geometric Brownian motion under risk-neutral measure
- **Terminal Sampling**: Efficient terminal value simulation (default)
- **Path Simulation**: Optional path-by-path simulation for extensibility
- **Variance Reduction**: Optional antithetic variates
- **Uncertainty Quantification**: Standard errors and 95% confidence intervals
- **Reproducibility**: Deterministic results via RNG seed
- **REST API**: FastAPI endpoint for programmatic access

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Python API

```python
from mcop import price_option

# Price a call option
result = price_option(
    S0=100.0,           # Spot price
    K=100.0,            # Strike price
    r=0.05,             # Risk-free rate (5%)
    sigma=0.2,          # Volatility (20%)
    T=1.0,              # Time to maturity (1 year)
    option_type="call", # Option type
    N=100000,           # Number of simulations
    seed=42,            # Random seed for reproducibility
)

print(f"Price: ${result.price:.4f}")
print(f"Standard Error: ${result.standard_error:.4f}")
print(f"95% CI: [${result.confidence_interval_95[0]:.4f}, ${result.confidence_interval_95[1]:.4f}]")
```

### REST API

Start the server:

```bash
uvicorn mcop.api:app --reload
```

Then make a POST request to `/price`:

```bash
curl -X POST "http://localhost:8000/price" \
  -H "Content-Type: application/json" \
  -d '{
    "S0": 100.0,
    "K": 100.0,
    "r": 0.05,
    "sigma": 0.2,
    "T": 1.0,
    "option_type": "call",
    "N": 100000,
    "seed": 42
  }'
```

## API Reference

### `price_option()`

Price a European vanilla option using Monte Carlo simulation.

**Parameters:**

- `S0` (float): Spot price (required)
- `K` (float): Strike price (required)
- `r` (float): Risk-free rate, annual, continuously compounded (required)
- `sigma` (float): Volatility, annual (required)
- `T` (float): Time to maturity in years (required)
- `option_type` (str): `"call"` or `"put"` (required)
- `N` (int): Number of simulations (required)
- `q` (float): Dividend yield, default 0.0 (optional)
- `seed` (int): Random number generator seed (optional)
- `antithetic` (bool): Use antithetic variates, default False (optional)
- `mode` (str): `"terminal"` or `"path"`, default `"terminal"` (optional)
- `M` (int): Number of time steps for path simulation, required if `mode="path"` (optional)

**Returns:**

`PricingResult` object with:
- `price` (float): Estimated option price
- `standard_error` (float): Standard error of the estimate
- `confidence_interval_95` (tuple): 95% confidence interval [lower, upper]
- `runtime_ms` (float): Runtime in milliseconds
- `diagnostics` (dict): Diagnostic information (N, mode, antithetic, etc.)

## Model

The pricer uses risk-neutral geometric Brownian motion:

```
dS = (r - q) S dt + sigma S dW
```

### Terminal Sampling (default)

For terminal sampling, the stock price at maturity is:

```
S_T = S0 * exp((r - q - 0.5*sigma²)*T + sigma*√T*Z)
```

where Z ~ N(0,1).

### Path Simulation

For path simulation, the stock price evolves as:

```
S_{t+Δt} = S_t * exp((r - q - 0.5*sigma²)*Δt + sigma*√Δt*Z)
```

where Δt = T / M.

## Variance Reduction

### Antithetic Variates

When `antithetic=True`, for each random draw Z, the pricer also uses -Z. This reduces variance by exploiting the symmetry of the normal distribution.

## Testing

Run tests with:

```bash
pytest tests/
```

Tests verify:
- Convergence to Black-Scholes prices
- Confidence interval scaling (width ∝ 1/√N)
- Edge cases (T=0, sigma=0, etc.)
- Reproducibility with seeds
- Input validation

## Examples

### Basic Call Option

```python
result = price_option(
    S0=100, K=100, r=0.05, sigma=0.2, T=1.0,
    option_type="call", N=100000, seed=42
)
```

### Put Option with Dividends

```python
result = price_option(
    S0=100, K=100, r=0.05, sigma=0.2, T=1.0,
    option_type="put", N=100000, q=0.02, seed=42
)
```

### With Antithetic Variates

```python
result = price_option(
    S0=100, K=100, r=0.05, sigma=0.2, T=1.0,
    option_type="call", N=100000, antithetic=True, seed=42
)
```

### Path Simulation

```python
result = price_option(
    S0=100, K=100, r=0.05, sigma=0.2, T=1.0,
    option_type="call", N=100000, mode="path", M=252, seed=42
)
```

## Project Structure

```
mcop/
├── mcop/              # Main package
│   ├── __init__.py    # Package initialization
│   ├── pricer.py      # Core pricing engine
│   └── api.py         # FastAPI REST endpoint
├── tests/             # Test suite
│   ├── __init__.py
│   └── test_pricer.py # Unit tests
├── requirements.txt   # Python dependencies
├── README.md          # This file
└── .gitignore         # Git ignore rules
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_pricer.py
```

## License

MIT

