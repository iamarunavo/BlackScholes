# 🚀 Black-Scholes Interactive Dashboard
Created By: Arunavo Chowdhury, Yaseen Choudhury
A comprehensive **Streamlit web application** for calculating European Call and Put option prices using the Black-Scholes formula, featuring interactive visualizations, real-time market data integration, and educational content.

## ✨ Features

### 🎯 Core Functionality
- **Black-Scholes Option Pricing**: Calculate European Call and Put option prices with high precision
- **Live Market Data**: Fetch real-time stock prices using Yahoo Finance API
- **Interactive Dashboard**: Real-time parameter adjustment with immediate visualization updates
- **Comprehensive Greeks Analysis**: Calculate Delta, Gamma, Theta, Vega, and Rho with detailed explanations

### 📊 Advanced Visualizations
- **Price Sensitivity Charts**: Interactive plots showing option prices vs spot price and volatility
- **Heatmap Analysis**: 2D heatmaps displaying option prices across different spot prices and volatilities
- **Real-time Metrics**: Live display of current parameters and calculated option prices
- **Professional UI**: Modern, responsive design with intuitive navigation

### 📚 Educational Content
- **Step-by-Step Derivation**: Complete mathematical derivation of the Black-Scholes formula
- **Interactive Learning**: Explanations of key concepts with real-time examples
- **Put-Call Parity Verification**: Mathematical validation of calculations
- **Greeks Explanations**: Detailed explanations of what each Greek represents

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository:**
```bash
git clone <repository-url>
cd BlackScholes
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 🚀 Running the Application

### Web Application (Recommended)
```bash
streamlit run app.py
```
The application will open in your default web browser at `http://localhost:8501`.

### Standalone Functions
```bash
python black_scholes.py
```
This will run the standalone Black-Scholes functions with example calculations.

## 📁 Project Structure

```
BlackScholes/
├── app.py                 # Main Streamlit web application (617 lines)
├── black_scholes.py       # Standalone Black-Scholes functions (122 lines)
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .gitignore            # Git ignore file
└── .devcontainer/        # Development container configuration
```

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **streamlit** | ≥1.28.0 | Web application framework |
| **numpy** | ≥1.24.0 | Numerical computing |
| **matplotlib** | ≥3.7.0 | Plotting and visualization |
| **scipy** | ≥1.10.0 | Scientific computing (normal distribution) |
| **seaborn** | ≥0.12.0 | Statistical data visualization |
| **yfinance** | ≥0.2.0 | Real-time financial data |

## 🧮 Black-Scholes Formula Implementation

The application implements the standard Black-Scholes formula for European options:

### Mathematical Foundation

**Call Option Price:**
```
C = S · N(d₁) - K · e^(-rT) · N(d₂)
```

**Put Option Price:**
```
P = K · e^(-rT) · N(-d₂) - S · N(-d₁)
```

**Where:**
- `d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)`
- `d₂ = d₁ - σ√T`
- `S` = Current stock price (Spot Price)
- `K` = Strike price
- `T` = Time to maturity (years)
- `r` = Risk-free interest rate
- `σ` = Volatility
- `N(·)` = Cumulative standard normal distribution

## 💡 Usage Examples

### Using the Standalone Functions

```python
from black_scholes import black_scholes, black_scholes_greeks

# Example parameters
S = 100  # Spot price
K = 100  # Strike price
T = 1.0  # Time to maturity (1 year)
r = 0.05 # Risk-free rate (5%)
sigma = 0.2  # Volatility (20%)

# Calculate option prices
call_price = black_scholes(S, K, T, r, sigma, "call")
put_price = black_scholes(S, K, T, r, sigma, "put")

print(f"Call Option Price: ${call_price:.4f}")
print(f"Put Option Price: ${put_price:.4f}")

# Calculate Greeks
greeks = black_scholes_greeks(S, K, T, r, sigma, "call")
print(f"Delta: {greeks['delta']:.4f}")
print(f"Gamma: {greeks['gamma']:.6f}")
print(f"Theta: {greeks['theta']:.4f}")
print(f"Vega: {greeks['vega']:.4f}")
```

### Web Application Features

1. **Interactive Dashboard Tab:**
   - Real-time parameter adjustment
   - Live option price calculations
   - Sensitivity analysis charts
   - Heatmap visualizations
   - Comprehensive Greeks analysis

2. **Educational Tab:**
   - Complete mathematical derivation
   - Step-by-step formula explanation
   - Interactive examples with current parameters
   - Put-call parity verification

## 🧪 Testing & Validation

### Run Built-in Tests
```bash
python black_scholes.py
```

**Expected Output:**
```
Call Option Price: $10.4506
Put Option Price: $5.5735

Call Option Greeks:
Price: $10.4506
Delta: 0.6368
Gamma: 0.0199
Theta: -6.4147
Vega: 37.8019

Put-Call Parity Check:
Call - Put: $4.8771
S - K*exp(-r*T): $4.8771
Difference: $0.000000
```

### Manual Testing
The application includes comprehensive error handling and input validation for:
- Invalid parameter ranges
- Division by zero scenarios
- Market data fetch failures
- Mathematical edge cases

## 📈 Real-World Applications

This tool is perfect for:
- **Options Traders**: Quick price calculations and Greeks analysis
- **Financial Analysts**: Risk assessment and sensitivity analysis
- **Students**: Learning quantitative finance concepts
- **Researchers**: Prototyping and testing option pricing models

---

*Built using Streamlit, NumPy, and modern Python libraries*
