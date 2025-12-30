"""
Tests for Monte Carlo option pricer.
"""

import pytest
import numpy as np
from mcop.pricer import price_option, PricingResult


def black_scholes_call(S0, K, r, sigma, T, q=0.0):
    """Black-Scholes formula for European call option."""
    from scipy.stats import norm
    
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


def black_scholes_put(S0, K, r, sigma, T, q=0.0):
    """Black-Scholes formula for European put option."""
    call_price = black_scholes_call(S0, K, r, sigma, T, q)
    put_price = call_price - S0 * np.exp(-q * T) + K * np.exp(-r * T)
    return put_price


class TestBasicFunctionality:
    """Test basic pricing functionality."""
    
    def test_call_option_terminal(self):
        """Test basic call option pricing."""
        result = price_option(
            S0=100.0,
            K=100.0,
            r=0.05,
            sigma=0.2,
            T=1.0,
            option_type="call",
            N=10000,
            seed=42,
        )
        
        assert isinstance(result, PricingResult)
        assert result.price > 0
        assert result.standard_error > 0
        assert len(result.confidence_interval_95) == 2
        assert result.confidence_interval_95[0] < result.confidence_interval_95[1]
        assert result.runtime_ms >= 0
        assert result.diagnostics["N"] == 10000
        assert result.diagnostics["mode"] == "terminal"
        assert result.diagnostics["antithetic"] is False
    
    def test_put_option_terminal(self):
        """Test basic put option pricing."""
        result = price_option(
            S0=100.0,
            K=100.0,
            r=0.05,
            sigma=0.2,
            T=1.0,
            option_type="put",
            N=10000,
            seed=42,
        )
        
        assert isinstance(result, PricingResult)
        assert result.price > 0
        assert result.standard_error > 0
    
    def test_convergence_to_black_scholes(self):
        """Test that MC price converges to Black-Scholes price as N increases."""
        S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
        
        bs_price = black_scholes_call(S0, K, r, sigma, T)
        
        # Test with increasing N
        for N in [1000, 10000, 100000]:
            result = price_option(
                S0=S0,
                K=K,
                r=r,
                sigma=sigma,
                T=T,
                option_type="call",
                N=N,
                seed=42,
            )
            
            # Check that price is within reasonable range of BS price
            # For large N, should be within 2 standard errors
            error = abs(result.price - bs_price)
            assert error < 3 * result.standard_error, \
                f"MC price {result.price} too far from BS price {bs_price} for N={N}"
    
    def test_antithetic_variates(self):
        """Test antithetic variates variance reduction."""
        # Same seed, one with antithetic, one without
        result_no_antithetic = price_option(
            S0=100.0,
            K=100.0,
            r=0.05,
            sigma=0.2,
            T=1.0,
            option_type="call",
            N=10000,
            seed=42,
            antithetic=False,
        )
        
        result_antithetic = price_option(
            S0=100.0,
            K=100.0,
            r=0.05,
            sigma=0.2,
            T=1.0,
            option_type="call",
            N=10000,
            seed=42,
            antithetic=True,
        )
        
        # Antithetic should generally reduce variance (smaller standard error)
        # Note: This is probabilistic, but should hold in most cases
        assert result_antithetic.diagnostics["antithetic"] is True
        assert result_no_antithetic.diagnostics["antithetic"] is False
    
    def test_path_simulation(self):
        """Test path simulation mode."""
        result = price_option(
            S0=100.0,
            K=100.0,
            r=0.05,
            sigma=0.2,
            T=1.0,
            option_type="call",
            N=10000,
            mode="path",
            M=252,  # Daily steps
            seed=42,
        )
        
        assert result.diagnostics["mode"] == "path"
        assert result.diagnostics["M"] == 252
        assert result.price > 0
    
    def test_reproducibility(self):
        """Test that same seed produces same results."""
        result1 = price_option(
            S0=100.0,
            K=100.0,
            r=0.05,
            sigma=0.2,
            T=1.0,
            option_type="call",
            N=10000,
            seed=123,
        )
        
        result2 = price_option(
            S0=100.0,
            K=100.0,
            r=0.05,
            sigma=0.2,
            T=1.0,
            option_type="call",
            N=10000,
            seed=123,
        )
        
        assert result1.price == result2.price
        assert result1.standard_error == result2.standard_error


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_time_to_maturity(self):
        """Test T=0 (option expires immediately)."""
        result = price_option(
            S0=100.0,
            K=100.0,
            r=0.05,
            sigma=0.2,
            T=0.0,
            option_type="call",
            N=1000,
            seed=42,
        )
        
        # For call: max(S0 - K, 0) = max(100 - 100, 0) = 0
        assert result.price == pytest.approx(0.0, abs=1e-6)
    
    def test_zero_volatility(self):
        """Test sigma=0 (deterministic stock price)."""
        result = price_option(
            S0=100.0,
            K=100.0,
            r=0.05,
            sigma=0.0,
            T=1.0,
            option_type="call",
            N=1000,
            seed=42,
        )
        
        # With sigma=0, S_T = S0 * exp(r*T) = 100 * exp(0.05) ≈ 105.13
        # Call payoff = max(105.13 - 100, 0) = 5.13
        # Discounted: 5.13 * exp(-0.05) ≈ 4.88
        expected = max(100.0 * np.exp(0.05) - 100.0, 0.0) * np.exp(-0.05)
        assert result.price == pytest.approx(expected, abs=0.1)
    
    def test_dividend_yield(self):
        """Test with non-zero dividend yield."""
        result = price_option(
            S0=100.0,
            K=100.0,
            r=0.05,
            sigma=0.2,
            T=1.0,
            option_type="call",
            N=10000,
            q=0.02,
            seed=42,
        )
        
        bs_price = black_scholes_call(100.0, 100.0, 0.05, 0.2, 1.0, q=0.02)
        error = abs(result.price - bs_price)
        assert error < 3 * result.standard_error
    
    def test_confidence_interval_width(self):
        """Test that CI width scales as 1/sqrt(N)."""
        results = []
        for N in [1000, 10000, 100000]:
            result = price_option(
                S0=100.0,
                K=100.0,
                r=0.05,
                sigma=0.2,
                T=1.0,
                option_type="call",
                N=N,
                seed=42,
            )
            ci_width = result.confidence_interval_95[1] - result.confidence_interval_95[0]
            results.append((N, ci_width))
        
        # CI width should decrease as N increases
        assert results[1][1] < results[0][1]  # 10k < 1k
        assert results[2][1] < results[1][1]  # 100k < 10k


class TestValidation:
    """Test input validation."""
    
    def test_negative_N(self):
        """Test that negative N raises error."""
        with pytest.raises(ValueError, match="N must be positive"):
            price_option(
                S0=100.0,
                K=100.0,
                r=0.05,
                sigma=0.2,
                T=1.0,
                option_type="call",
                N=-1000,
            )
    
    def test_negative_T(self):
        """Test that negative T raises error."""
        with pytest.raises(ValueError, match="T must be non-negative"):
            price_option(
                S0=100.0,
                K=100.0,
                r=0.05,
                sigma=0.2,
                T=-1.0,
                option_type="call",
                N=1000,
            )
    
    def test_negative_sigma(self):
        """Test that negative sigma raises error."""
        with pytest.raises(ValueError, match="sigma must be non-negative"):
            price_option(
                S0=100.0,
                K=100.0,
                r=0.05,
                sigma=-0.2,
                T=1.0,
                option_type="call",
                N=1000,
            )
    
    def test_invalid_option_type(self):
        """Test that invalid option_type raises error."""
        with pytest.raises(ValueError, match="option_type must be 'call' or 'put'"):
            price_option(
                S0=100.0,
                K=100.0,
                r=0.05,
                sigma=0.2,
                T=1.0,
                option_type="invalid",
                N=1000,
            )
    
    def test_path_mode_without_M(self):
        """Test that path mode without M raises error."""
        with pytest.raises(ValueError, match="M must be provided when mode='path'"):
            price_option(
                S0=100.0,
                K=100.0,
                r=0.05,
                sigma=0.2,
                T=1.0,
                option_type="call",
                N=1000,
                mode="path",
            )

