import numpy as np
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, option_type):
    """
    Calculate European Call and Put option prices using the Black-Scholes formula.
    
    Parameters:
    S (float): Current stock price (Spot Price)
    K (float): Strike price
    T (float): Time to maturity in years
    r (float): Risk-free interest rate (annual)
    sigma (float): Volatility (annual)
    option_type (str): "call" or "put"
    
    Returns:
    float: Option price
    
    Example:
    >>> black_scholes(100, 100, 1, 0.05, 0.2, "call")
    10.450583572185565
    >>> black_scholes(100, 100, 1, 0.05, 0.2, "put")
    5.573526022256971
    """
    # Calculate d1 and d2
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == "call":
        # Call option price
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        # Put option price
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    return price

def black_scholes_greeks(S, K, T, r, sigma, option_type):
    """
    Calculate option price and Greeks (Delta, Gamma, Theta, Vega) using Black-Scholes.
    
    Parameters:
    S (float): Current stock price (Spot Price)
    K (float): Strike price
    T (float): Time to maturity in years
    r (float): Risk-free interest rate (annual)
    sigma (float): Volatility (annual)
    option_type (str): "call" or "put"
    
    Returns:
    dict: Dictionary containing price and Greeks
    """
    # Calculate d1 and d2
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calculate option price
    price = black_scholes(S, K, T, r, sigma, option_type)
    
    # Calculate Greeks
    if option_type.lower() == "call":
        delta = norm.cdf(d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                r * K * np.exp(-r * T) * norm.cdf(d2))
    elif option_type.lower() == "put":
        delta = norm.cdf(d1) - 1
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                r * K * np.exp(-r * T) * norm.cdf(-d2))
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    # Gamma is the same for both call and put
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Vega is the same for both call and put
    vega = S * np.sqrt(T) * norm.pdf(d1)
    
    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega
    }

# Example usage and testing
if __name__ == "__main__":
    # Test parameters
    S = 100  # Spot price
    K = 100  # Strike price
    T = 1.0  # Time to maturity (1 year)
    r = 0.05  # Risk-free rate (5%)
    sigma = 0.2  # Volatility (20%)
    
    # Calculate call option price
    call_price = black_scholes(S, K, T, r, sigma, "call")
    print(f"Call Option Price: ${call_price:.4f}")
    
    # Calculate put option price
    put_price = black_scholes(S, K, T, r, sigma, "put")
    print(f"Put Option Price: ${put_price:.4f}")
    
    # Calculate Greeks for call option
    call_greeks = black_scholes_greeks(S, K, T, r, sigma, "call")
    print(f"\nCall Option Greeks:")
    print(f"Price: ${call_greeks['price']:.4f}")
    print(f"Delta: {call_greeks['delta']:.4f}")
    print(f"Gamma: {call_greeks['gamma']:.4f}")
    print(f"Theta: {call_greeks['theta']:.4f}")
    print(f"Vega: {call_greeks['vega']:.4f}")
    
    # Verify put-call parity
    # C - P = S - K*exp(-r*T)
    put_call_parity = call_price - put_price
    theoretical_parity = S - K * np.exp(-r * T)
    print(f"\nPut-Call Parity Check:")
    print(f"Call - Put: ${put_call_parity:.4f}")
    print(f"S - K*exp(-r*T): ${theoretical_parity:.4f}")
    print(f"Difference: ${abs(put_call_parity - theoretical_parity):.6f}")
