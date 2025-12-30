"""
Monte Carlo Option Pricer

Prices European vanilla options using Monte Carlo simulation
under Black-Scholes assumptions.
"""

import numpy as np
import time
from typing import Literal, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class PricingResult:
    """Result of Monte Carlo option pricing."""
    price: float
    standard_error: float
    confidence_interval_95: tuple[float, float]
    runtime_ms: float
    diagnostics: Dict[str, Any]


def price_option(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: Literal["call", "put"],
    N: int,
    q: float = 0.0,
    seed: Optional[int] = None,
    antithetic: bool = False,
    mode: Literal["terminal", "path"] = "terminal",
    M: Optional[int] = None,
) -> PricingResult:
    """
    Price a European vanilla option using Monte Carlo simulation.
    
    Parameters
    ----------
    S0 : float
        Spot price
    K : float
        Strike price
    r : float
        Risk-free rate (annual, continuously compounded)
    sigma : float
        Volatility (annual)
    T : float
        Time to maturity (years)
    option_type : {"call", "put"}
        Option type
    N : int
        Number of simulations
    q : float, default 0.0
        Dividend yield (annual, continuously compounded)
    seed : int, optional
        Random number generator seed for reproducibility
    antithetic : bool, default False
        Use antithetic variates for variance reduction
    mode : {"terminal", "path"}, default "terminal"
        Simulation mode: terminal sampling or path simulation
    M : int, optional
        Number of time steps for path simulation (required if mode="path")
    
    Returns
    -------
    PricingResult
        Pricing result with price, standard error, confidence interval,
        runtime, and diagnostics
    """
    start_time = time.perf_counter()
    
    # Validation
    if N <= 0:
        raise ValueError("N must be positive")
    if T < 0:
        raise ValueError("T must be non-negative")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if S0 <= 0:
        raise ValueError("S0 must be positive")
    if K <= 0:
        raise ValueError("K must be positive")
    if option_type not in ["call", "put"]:
        raise ValueError("option_type must be 'call' or 'put'")
    if mode == "path" and M is None:
        raise ValueError("M must be provided when mode='path'")
    if mode == "path" and M <= 0:
        raise ValueError("M must be positive")
    
    # Set random seed
    if seed is not None:
        np.random.seed(seed)
    
    # Determine number of random draws
    if antithetic:
        n_draws = N // 2
    else:
        n_draws = N
    
    # Generate terminal stock prices
    if mode == "terminal":
        ST = _simulate_terminal(S0, r, q, sigma, T, n_draws, antithetic)
    else:  # mode == "path"
        ST = _simulate_path(S0, r, q, sigma, T, M, n_draws, antithetic)
    
    # Calculate payoffs
    if option_type == "call":
        payoffs = np.maximum(ST - K, 0.0)
    else:  # put
        payoffs = np.maximum(K - ST, 0.0)
    
    # Discount to present value
    discounted_payoffs = np.exp(-r * T) * payoffs
    
    # Calculate statistics
    price = np.mean(discounted_payoffs)
    std_payoffs = np.std(discounted_payoffs, ddof=1)  # Sample std
    standard_error = std_payoffs / np.sqrt(len(discounted_payoffs))
    
    # 95% confidence interval (normal approximation)
    z_95 = 1.96
    ci_lower = price - z_95 * standard_error
    ci_upper = price + z_95 * standard_error
    confidence_interval_95 = (ci_lower, ci_upper)
    
    # Runtime
    runtime_ms = (time.perf_counter() - start_time) * 1000
    
    # Diagnostics
    diagnostics = {
        "N": N,
        "mode": mode,
        "antithetic": antithetic,
    }
    if mode == "path":
        diagnostics["M"] = M
    
    return PricingResult(
        price=float(price),
        standard_error=float(standard_error),
        confidence_interval_95=confidence_interval_95,
        runtime_ms=runtime_ms,
        diagnostics=diagnostics,
    )


def _simulate_terminal(
    S0: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    n_draws: int,
    antithetic: bool,
) -> np.ndarray:
    """
    Simulate terminal stock prices using terminal sampling.
    
    S_T = S0 * exp((r - q - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
    where Z ~ N(0,1)
    """
    # Generate standard normal random variables
    Z = np.random.standard_normal(n_draws)
    
    # Apply antithetic variates if requested
    if antithetic:
        Z = np.concatenate([Z, -Z])
    
    # Calculate drift and diffusion terms
    drift = (r - q - 0.5 * sigma ** 2) * T
    diffusion = sigma * np.sqrt(T) * Z
    
    # Terminal stock prices
    ST = S0 * np.exp(drift + diffusion)
    
    return ST


def _simulate_path(
    S0: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    M: int,
    n_draws: int,
    antithetic: bool,
) -> np.ndarray:
    """
    Simulate terminal stock prices using path simulation.
    
    S_{t+Δt} = S_t * exp((r - q - 0.5*sigma^2)*Δt + sigma*sqrt(Δt)*Z)
    where Δt = T / M
    """
    dt = T / M
    drift_term = (r - q - 0.5 * sigma ** 2) * dt
    diffusion_term = sigma * np.sqrt(dt)
    
    # Generate all random numbers upfront
    # Shape: (M, n_draws) - one row per time step, one column per path
    Z_all = np.random.standard_normal((M, n_draws))
    
    # Initialize stock prices
    S = np.full(n_draws, S0, dtype=float)
    
    # Simulate each time step
    for t in range(M):
        Z = Z_all[t, :]
        S = S * np.exp(drift_term + diffusion_term * Z)
    
    # Apply antithetic variates if requested
    if antithetic:
        # Simulate antithetic paths using negated random variables
        S_antithetic = np.full(n_draws, S0, dtype=float)
        for t in range(M):
            Z = -Z_all[t, :]  # Use negated random variables
            S_antithetic = S_antithetic * np.exp(drift_term + diffusion_term * Z)
        S = np.concatenate([S, S_antithetic])
    
    return S

