"""
FastAPI endpoint for Monte Carlo option pricing.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
from typing import Literal, Optional
from .pricer import price_option, PricingResult

app = FastAPI(
    title="Monte Carlo Option Pricer",
    description="Price European vanilla options using Monte Carlo simulation",
    version="1.0.0",
)


class PricingRequest(BaseModel):
    """Request model for option pricing."""
    S0: float = Field(..., gt=0, description="Spot price")
    K: float = Field(..., gt=0, description="Strike price")
    r: float = Field(..., description="Risk-free rate (annual, continuously compounded)")
    sigma: float = Field(..., ge=0, description="Volatility (annual)")
    T: float = Field(..., ge=0, description="Time to maturity (years)")
    option_type: Literal["call", "put"] = Field(..., description="Option type")
    N: int = Field(..., gt=0, description="Number of simulations")
    q: float = Field(0.0, description="Dividend yield (annual, continuously compounded)")
    seed: Optional[int] = Field(None, description="Random number generator seed")
    antithetic: bool = Field(False, description="Use antithetic variates")
    mode: Literal["terminal", "path"] = Field("terminal", description="Simulation mode")
    M: Optional[int] = Field(None, gt=0, description="Number of time steps (required if mode='path')")
    
    @model_validator(mode="after")
    def validate_path_mode(self):
        """Validate that M is provided when mode is 'path'."""
        if self.mode == "path" and self.M is None:
            raise ValueError("M must be provided when mode='path'")
        return self


class PricingResponse(BaseModel):
    """Response model for option pricing."""
    price: float
    standard_error: float
    confidence_interval_95: list[float]
    runtime_ms: float
    diagnostics: dict


@app.post("/price", response_model=PricingResponse)
async def price(request: PricingRequest):
    """
    Price a European vanilla option using Monte Carlo simulation.
    """
    try:
        result: PricingResult = price_option(
            S0=request.S0,
            K=request.K,
            r=request.r,
            sigma=request.sigma,
            T=request.T,
            option_type=request.option_type,
            N=request.N,
            q=request.q,
            seed=request.seed,
            antithetic=request.antithetic,
            mode=request.mode,
            M=request.M,
        )
        
        return PricingResponse(
            price=result.price,
            standard_error=result.standard_error,
            confidence_interval_95=list(result.confidence_interval_95),
            runtime_ms=result.runtime_ms,
            diagnostics=result.diagnostics,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Monte Carlo Option Pricer API",
        "version": "1.0.0",
        "endpoints": {
            "POST /price": "Price an option using Monte Carlo simulation"
        }
    }

