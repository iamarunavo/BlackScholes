# Black-Scholes Option Pricing Calculator

A Streamlit application for calculating European Call and Put option prices using the Black-Scholes formula, with interactive visualizations and Greeks calculations.

## Features

- **Black-Scholes Option Pricing**: Calculate European Call and Put option prices
- **Interactive Interface**: Adjust parameters in real-time using Streamlit widgets
- **Visualization**: Plot option prices vs spot price with interactive charts
- **Greeks Calculation**: Compute Delta, Gamma, Theta, and Vega
- **Put-Call Parity Verification**: Validate calculations using put-call parity
- **Mathematical Formulas**: Display the Black-Scholes formulas with LaTeX

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the App

To run the Streamlit app, use the following command:
```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`.

## Project Structure

```
├── app.py                 # Main Streamlit application
├── black_scholes.py       # Standalone Black-Scholes functions
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── .gitignore            # Git ignore file
```

## Dependencies

- **streamlit**: Web application framework
- **numpy**: Numerical computing library
- **matplotlib**: Plotting and visualization
- **scipy**: Scientific computing library (for normal distribution)

## Black-Scholes Formula

The application implements the standard Black-Scholes formula for European options:

**Call Option Price:**
```
C = S * N(d₁) - K * e^(-rT) * N(d₂)
```

**Put Option Price:**
```
P = K * e^(-rT) * N(-d₂) - S * N(-d₁)
```

Where:
- d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
- d₂ = d₁ - σ√T
- S = Current stock price
- K = Strike price
- T = Time to maturity
- r = Risk-free interest rate
- σ = Volatility
- N(·) = Cumulative normal distribution function

## Usage Examples

### Using the Standalone Function

```python
from black_scholes import black_scholes

# Calculate call option price
call_price = black_scholes(100, 100, 1, 0.05, 0.2, "call")
print(f"Call Option Price: ${call_price:.4f}")

# Calculate put option price
put_price = black_scholes(100, 100, 1, 0.05, 0.2, "put")
print(f"Put Option Price: ${put_price:.4f}")
```

### Calculating Greeks

```python
from black_scholes import black_scholes_greeks

# Get price and Greeks
greeks = black_scholes_greeks(100, 100, 1, 0.05, 0.2, "call")
print(f"Price: ${greeks['price']:.4f}")
print(f"Delta: {greeks['delta']:.4f}")
print(f"Gamma: {greeks['gamma']:.4f}")
print(f"Theta: {greeks['theta']:.4f}")
print(f"Vega: {greeks['vega']:.4f}")
```

## Testing

Run the standalone Black-Scholes function to verify calculations:

```bash
python black_scholes.py
```

This will output:
- Call and Put option prices
- Greeks for the call option
- Put-call parity verification

## Customization

The Streamlit app includes:
- Interactive parameter inputs in the sidebar
- Real-time price calculations
- Option price vs spot price visualization
- Mathematical formula display
- Error handling for invalid inputs

Edit `app.py` to add additional features like:
- More Greeks calculations
- Different option types (American, Asian, etc.)
- Monte Carlo simulations
- Historical volatility calculations
