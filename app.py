import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
import seaborn as sns
import yfinance as yf

# Set page configuration
st.set_page_config(
    page_title="Black-Scholes Model",
    page_icon="🚀",
    layout="wide"
)

# --- Educational Add-ons: Derivation + Tooltips ----------------------------
def calculate_greeks(S, K, T, r, sigma):
    """
    Calculate option Greeks (Delta, Gamma, Theta, Vega, Rho)
    """
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Call option Greeks
    call_delta = norm.cdf(d1)
    call_gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    call_theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                  r * K * np.exp(-r*T) * norm.cdf(d2)) / 365  # Daily theta
    call_vega = S * np.sqrt(T) * norm.pdf(d1) / 100  # Per 1% change in volatility
    call_rho = K * T * np.exp(-r*T) * norm.cdf(d2) / 100  # Per 1% change in rate
    
    # Put option Greeks
    put_delta = call_delta - 1
    put_gamma = call_gamma
    put_theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                 r * K * np.exp(-r*T) * norm.cdf(-d2)) / 365
    put_vega = call_vega
    put_rho = -K * T * np.exp(-r*T) * norm.cdf(-d2) / 100
    
    return {
        'call': {'delta': call_delta, 'gamma': call_gamma, 'theta': call_theta, 
                'vega': call_vega, 'rho': call_rho},
        'put': {'delta': put_delta, 'gamma': put_gamma, 'theta': put_theta, 
               'vega': put_vega, 'rho': put_rho}
    }

def render_greeks_analysis(S, K, T, r, sigma):
    """
    Display Greeks analysis with explanations
    """
    greeks = calculate_greeks(S, K, T, r, sigma)
    
    st.markdown("## 📊 Greeks Analysis")
    st.markdown("Greeks measure the sensitivity of option prices to various factors:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Call Option Greeks")
        st.metric("Delta (Δ)", f"{greeks['call']['delta']:.4f}", 
                 help="Rate of change in option price per $1 change in stock price")
        st.metric("Gamma (Γ)", f"{greeks['call']['gamma']:.6f}", 
                 help="Rate of change in delta per $1 change in stock price")
        st.metric("Theta (Θ)", f"{greeks['call']['theta']:.6f}", 
                 help="Daily time decay of option value")
        st.metric("Vega (ν)", f"{greeks['call']['vega']:.4f}", 
                 help="Change in option price per 1% change in volatility")
        st.metric("Rho (ρ)", f"{greeks['call']['rho']:.4f}", 
                 help="Change in option price per 1% change in interest rate")
    
    with col2:
        st.subheader("📉 Put Option Greeks")
        st.metric("Delta (Δ)", f"{greeks['put']['delta']:.4f}", 
                 help="Rate of change in option price per $1 change in stock price")
        st.metric("Gamma (Γ)", f"{greeks['put']['gamma']:.6f}", 
                 help="Rate of change in delta per $1 change in stock price")
        st.metric("Theta (Θ)", f"{greeks['put']['theta']:.6f}", 
                 help="Daily time decay of option value")
        st.metric("Vega (ν)", f"{greeks['put']['vega']:.4f}", 
                 help="Change in option price per 1% change in volatility")
        st.metric("Rho (ρ)", f"{greeks['put']['rho']:.4f}", 
                 help="Change in option price per 1% change in interest rate")
    
    # Greeks explanation
    with st.expander("💡 What do Greeks mean?", expanded=False):
        st.markdown("""
        **Delta (Δ)**: How much the option price changes when the stock price changes by $1
        - Call: 0 to 1 (increases with stock price)
        - Put: -1 to 0 (decreases with stock price)
        
        **Gamma (Γ)**: How much delta changes when the stock price changes by $1
        - Highest when option is at-the-money
        - Measures convexity of option payoff
        
        **Theta (Θ)**: How much the option loses value each day due to time decay
        - Always negative (options lose value over time)
        - Accelerates as expiration approaches
        
        **Vega (ν)**: How much the option price changes when volatility changes by 1%
        - Higher volatility = higher option prices
        - At-the-money options are most sensitive to volatility
        
        **Rho (ρ)**: How much the option price changes when interest rates change by 1%
        - Calls benefit from higher rates, puts are hurt
        - Longer-term options are more sensitive
        """)

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

def get_live_price(ticker):
    """
    Fetch the latest market price for a given stock ticker.
    
    Parameters:
    ticker (str): Stock ticker symbol
    
    Returns:
    float: Latest closing price, or None if error occurs
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        if len(data) > 0:
            return round(data["Close"].iloc[-1], 2)
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

def plot_heatmap(bs_model, spot_range, vol_range, strike):
    """
    Generate heatmaps using seaborn for better visualization with numerical values
    """
    call_prices = np.zeros((len(vol_range), len(spot_range)))
    put_prices = np.zeros((len(vol_range), len(spot_range)))
    
    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            try:
                call_prices[i, j] = black_scholes(spot, strike, bs_model['T'], bs_model['r'], vol, "call")
                put_prices[i, j] = black_scholes(spot, strike, bs_model['T'], bs_model['r'], vol, "put")
            except:
                call_prices[i, j] = np.nan
                put_prices[i, j] = np.nan
    
    # Plotting Call Price Heatmap
    fig_call, ax_call = plt.subplots(figsize=(10, 8))
    sns.heatmap(call_prices, 
                xticklabels=np.round(spot_range, 2), 
                yticklabels=np.round(vol_range, 2), 
                annot=True, 
                fmt=".2f", 
                cmap="viridis", 
                ax=ax_call)
    ax_call.set_title('CALL', fontsize=16, fontweight='bold')
    ax_call.set_xlabel('Spot Price')
    ax_call.set_ylabel('Volatility')
    
    # Plotting Put Price Heatmap
    fig_put, ax_put = plt.subplots(figsize=(10, 8))
    sns.heatmap(put_prices, 
                xticklabels=np.round(spot_range, 2), 
                yticklabels=np.round(vol_range, 2), 
                annot=True, 
                fmt=".2f", 
                cmap="plasma", 
                ax=ax_put)
    ax_put.set_title('PUT', fontsize=16, fontweight='bold')
    ax_put.set_xlabel('Spot Price')
    ax_put.set_ylabel('Volatility')
    
    return fig_call, fig_put

# Main app
def main():
    
    try:
        import matplotlib
        matplotlib.use('Agg')  
    except ImportError:
        st.error("Matplotlib is not installed. Please install it with: pip install matplotlib")
        return
    
    st.title("🚀 Black-Scholes Interactive Dashboard")
    st.markdown("---")
    
    # Sidebar - Settings Panel
    with st.sidebar:
        st.title("🚀 Black-Scholes Model")
        st.markdown("---")
        st.write("**Created by:** Arunavo Chowdhury, Yaseen Choudhury")
        st.markdown("---")
        
        st.header("⚙️ Settings Panel")
        
        # Market data section
        st.subheader("📊 Market Data")
        ticker = st.text_input("Stock Ticker (Optional)", value="", 
                              help="Enter stock symbol (e.g., AAPL) to fetch live prices")
        
        # Fetch live price if ticker provided
        live_price = None
        if ticker:
            live_price = get_live_price(ticker)
            if live_price:
                st.success(f"📈 {ticker}: ${live_price}")
                st.info("Live price will be used as default for Spot Price")
            else:
                st.warning(f"⚠️ Unable to fetch {ticker} price")
        
        st.markdown("---")
        
        # Model parameters section
        st.subheader("📐 Model Parameters")
        
        # Spot Price with live price integration
        default_spot = live_price if live_price else 100.0
        S = st.number_input("Spot Price (S)", 
                           min_value=0.01, 
                           value=default_spot, 
                           step=0.01,
                           help="The current market price of the underlying asset.")
        
        K = st.number_input("Strike Price (K)", 
                           min_value=0.01, 
                           value=100.0, 
                           step=0.01,
                           help="The price at which the option can be exercised at expiration.")
        
        sigma = st.slider("Volatility (σ)", 
                         min_value=0.01, 
                         max_value=1.0, 
                         value=0.20, 
                         step=0.01,
                         help="How much the stock price fluctuates, expressed as a percentage.")
        
        T = st.slider("Time to Maturity (T)", 
                     min_value=0.01, 
                     max_value=5.0, 
                     value=1.0, 
                     step=0.01,
                     help="Time until option expiration in years.")
        
        r = st.slider("Risk-Free Interest Rate (r)", 
                     min_value=0.0, 
                     max_value=0.20, 
                     value=0.05, 
                     step=0.001,
                     help="Annual risk-free interest rate (continuously compounded).")
        

    
    # Create tabs
    tab1, tab2 = st.tabs(["📊 Dashboard", "📚 Education"])
    
    # Tab 1: Main Dashboard
    with tab1:
        render_main_dashboard(S, K, T, r, sigma, ticker, live_price)
    
    # Tab 2: Education
    with tab2:
        render_education_tab(S, K, T, r, sigma)

def render_main_dashboard(S, K, T, r, sigma, ticker, live_price):
    """Render the main dashboard with charts and analysis"""
    st.header("📊 Black-Scholes Interactive Dashboard")
    
    # Current parameters display
    st.subheader("📋 Current Parameters")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Spot Price (S)", f"${S:.2f}")
    with col2:
        st.metric("Strike Price (K)", f"${K:.2f}")
    with col3:
        st.metric("Volatility (σ)", f"{sigma:.1%}")
    with col4:
        st.metric("Time to Maturity (T)", f"{T:.2f} years")
    with col5:
        st.metric("Risk-Free Rate (r)", f"{r:.1%}")
    
    st.markdown("---")
    
    # Option prices calculation and display
    st.subheader("💰 Option Prices")
    try:
        call_price = black_scholes(S, K, T, r, sigma, "call")
        put_price = black_scholes(S, K, T, r, sigma, "put")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div style="background-color: #00ff88; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid #00cc6a;">
                <h2 style="color: #000000; margin: 0;">📈 CALL Option</h2>
                <h1 style="color: #000000; margin: 10px 0;">${call_price:.2f}</h1>
                <p style="margin: 0; font-size: 14px;">Black-Scholes Price</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background-color: #ff69b4; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid #cc4a8a;">
                <h2 style="color: #000000; margin: 0;">📉 PUT Option</h2>
                <h1 style="color: #000000; margin: 10px 0;">${put_price:.2f}</h1>
                <p style="margin: 0; font-size: 14px;">Black-Scholes Price</p>
            </div>
            """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error calculating option prices: {str(e)}")
    
    st.markdown("---")
    
    # Interactive Charts
    st.subheader("📈 Interactive Price Charts")
    
    # Chart type selector
    chart_type = st.selectbox("Choose Chart Type:", 
                             ["Call & Put Price Sensitivity", "Heatmap Analysis"],
                             help="Select the type of visualization you want to see")
    
    if chart_type == "Call & Put Price Sensitivity":
        render_sensitivity_charts(S, K, T, r, sigma)
    else:
        render_heatmap_analysis(S, K, T, r, sigma)
    
    st.markdown("---")
    
    # Sensitivity Analysis (Greeks) - after charts
    st.subheader("📊 Sensitivity Analysis (Greeks)")
    render_greeks_analysis(S, K, T, r, sigma)

def render_sensitivity_charts(S, K, T, r, sigma):
    """Render sensitivity charts for call and put prices"""
    
    # Spot price sensitivity
    st.subheader("📈 Price Sensitivity to Spot Price")
    spot_range = np.linspace(S * 0.7, S * 1.3, 50)
    call_prices = [black_scholes(s, K, T, r, sigma, "call") for s in spot_range]
    put_prices = [black_scholes(s, K, T, r, sigma, "put") for s in spot_range]
    
    # Create the chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(spot_range, call_prices, label='Call Option', color='green', linewidth=2)
    ax.plot(spot_range, put_prices, label='Put Option', color='red', linewidth=2)
    ax.axvline(x=S, color='blue', linestyle='--', alpha=0.7, label=f'Current Price: ${S:.2f}')
    ax.axvline(x=K, color='orange', linestyle='--', alpha=0.7, label=f'Strike Price: ${K:.2f}')
    ax.set_xlabel('Spot Price ($)')
    ax.set_ylabel('Option Price ($)')
    ax.set_title('Option Price Sensitivity to Spot Price')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Volatility sensitivity
    st.subheader("📊 Price Sensitivity to Volatility")
    vol_range = np.linspace(0.05, 0.50, 50)
    call_prices_vol = [black_scholes(S, K, T, r, v, "call") for v in vol_range]
    put_prices_vol = [black_scholes(S, K, T, r, v, "put") for v in vol_range]
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(vol_range, call_prices_vol, label='Call Option', color='green', linewidth=2)
    ax2.plot(vol_range, put_prices_vol, label='Put Option', color='red', linewidth=2)
    ax2.axvline(x=sigma, color='blue', linestyle='--', alpha=0.7, label=f'Current Vol: {sigma:.1%}')
    ax2.set_xlabel('Volatility')
    ax2.set_ylabel('Option Price ($)')
    ax2.set_title('Option Price Sensitivity to Volatility')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    st.pyplot(fig2)

def render_heatmap_analysis(S, K, T, r, sigma):
    """Render heatmap analysis"""
    st.subheader("🔥 Heatmap Analysis")
    
    # Heatmap parameters
    col1, col2 = st.columns(2)
    with col1:
        min_spot = st.number_input("Min Spot Price", value=max(0.01, S * 0.8), step=0.01)
        max_spot = st.number_input("Max Spot Price", value=S * 1.2, step=0.01)
    with col2:
        min_vol = st.number_input("Min Volatility", value=0.10, min_value=0.01, max_value=1.00, step=0.01)
        max_vol = st.number_input("Max Volatility", value=0.30, min_value=0.01, max_value=1.00, step=0.01)
    
    # Create ranges for heatmap
    spot_range = np.linspace(min_spot, max_spot, 10)
    vol_range = np.linspace(min_vol, max_vol, 10)
    
    try:
        # Prepare model data for heatmap function
        bs_model = {'T': T, 'r': r}
        
        # Create side-by-side heatmaps
        col1, col2 = st.columns([1,1], gap="small")
        
        with col1:
            st.subheader("📈 Call Price Heatmap")
            heatmap_fig_call, _ = plot_heatmap(bs_model, spot_range, vol_range, K)
            st.pyplot(heatmap_fig_call)
        
        with col2:
            st.subheader("📉 Put Price Heatmap")
            _, heatmap_fig_put = plot_heatmap(bs_model, spot_range, vol_range, K)
            st.pyplot(heatmap_fig_put)
        
    except Exception as e:
        st.error(f"Error creating heatmaps: {str(e)}")

def render_education_tab(S, K, T, r, sigma):
    """Render the education tab with step-by-step derivation"""
    st.header("📚 Black-Scholes Formula: Step-by-Step Derivation")
    
    st.markdown("""
    This section explains the mathematical foundation of the Black-Scholes option pricing model.
    We'll walk through the key concepts and mathematical steps that lead to the final formula.
    """)
    
    # Step 1: Assumptions
    with st.expander("1️⃣ Key Assumptions", expanded=True):
        st.markdown("""
        The Black-Scholes model is based on several important assumptions:
        
        - **No Arbitrage**: Markets are efficient with no risk-free profit opportunities
        - **Frictionless Markets**: No transaction costs, taxes, or restrictions
        - **Continuous Trading**: Assets can be traded continuously
        - **Constant Parameters**: Volatility (σ) and risk-free rate (r) are constant
        - **Geometric Brownian Motion**: Stock prices follow a specific random process
        - **European Options**: Options can only be exercised at expiration
        """)
    
    # Step 2: Stock Price Dynamics
    with st.expander("2️⃣ Stock Price Dynamics", expanded=False):
        st.markdown("""
        The model assumes stock prices follow **Geometric Brownian Motion (GBM)**:
        """)
        st.latex(r"dS_t = \mu S_t\,dt + \sigma S_t\,dW_t")
        st.markdown("""
        Where:
        - $S_t$ = Stock price at time t
        - $\\mu$ = Expected return (drift)
        - $\\sigma$ = Volatility
        - $dW_t$ = Random Wiener process increment
        - $dt$ = Small time increment
        
        Under **risk-neutral pricing**, we replace $\\mu$ with $r$ (risk-free rate):
        """)
        st.latex(r"dS_t = r S_t\,dt + \sigma S_t\,dW_t")
    
    # Step 3: Option Value Function
    with st.expander("3️⃣ Option Value Function", expanded=False):
        st.markdown("""
        The option value $V(S,t)$ depends on the stock price $S$ and time $t$.
        Using **Itô's Lemma**, we can express how the option value changes:
        """)
        st.latex(r"""
        dV = \left(\frac{\partial V}{\partial t}
        + \mu S \frac{\partial V}{\partial S}
        + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}\right) dt
        + \sigma S \frac{\partial V}{\partial S} dW_t
        """)
        st.markdown("""
        This shows how the option value changes due to:
        - Time decay ($\\frac{\\partial V}{\\partial t}$)
        - Stock price changes ($\\frac{\\partial V}{\\partial S}$)
        - Volatility effects ($\\frac{\\partial^2 V}{\\partial S^2}$)
        """)
    
    # Step 4: Risk-Neutral Pricing
    with st.expander("4️⃣ Risk-Neutral Pricing", expanded=False):
        st.markdown("""
        The key insight is **risk-neutral pricing**. We construct a portfolio that eliminates randomness:
        
        **Portfolio**: Long 1 option + Short $\\Delta$ shares of stock
        
        The portfolio value is: $V - \\Delta S$
        
        To eliminate randomness, we choose $\\Delta = \\frac{\\partial V}{\\partial S}$ (this is called **delta hedging**).
        
        Since the portfolio is risk-free, it must earn the risk-free rate $r$:
        """)
        st.latex(r"""
        \frac{\partial V}{\partial t}
        + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}
        + r S \frac{\partial V}{\partial S}
        - rV = 0
        """)
        st.markdown("""
        This is the **Black-Scholes Partial Differential Equation (PDE)**.
        """)
    
    # Step 5: Boundary Conditions
    with st.expander("5️⃣ Boundary Conditions", expanded=False):
        st.markdown("""
        To solve the PDE, we need boundary conditions at expiration ($t = T$):
        
        **Call Option**: $V(S,T) = \\max(S-K, 0)$
        - If $S > K$: Option is worth $S-K$ (in-the-money)
        - If $S \\leq K$: Option is worthless (out-of-the-money)
        
        **Put Option**: $V(S,T) = \\max(K-S, 0)$
        - If $S < K$: Option is worth $K-S$ (in-the-money)
        - If $S \\geq K$: Option is worthless (out-of-the-money)
        """)
    
    # Step 6: Solution
    with st.expander("6️⃣ Closed-Form Solution", expanded=True):
        st.markdown("""
        The solution to the Black-Scholes PDE gives us the famous formulas:
        """)
        
        st.latex(r"""
        d_1 = \frac{\ln(S/K) + (r + \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}
        """)
        st.latex(r"""
        d_2 = d_1 - \sigma\sqrt{T}
        """)
        
        st.markdown("**Call Option Price:**")
        st.latex(r"""
        C = S \cdot N(d_1) - K \cdot e^{-rT} \cdot N(d_2)
        """)
        
        st.markdown("**Put Option Price:**")
        st.latex(r"""
        P = K \cdot e^{-rT} \cdot N(-d_2) - S \cdot N(-d_1)
        """)
        
        st.markdown("""
        Where:
        - $N(\\cdot)$ = Cumulative standard normal distribution
        - $S$ = Current stock price
        - $K$ = Strike price
        - $T$ = Time to maturity
        - $r$ = Risk-free interest rate
        - $\\sigma$ = Volatility
        """)
    
    # Step 7: Intuition
    with st.expander("7️⃣ Understanding the Formula", expanded=False):
        st.markdown("""
        **Call Option Intuition:**
        - $S \\cdot N(d_1)$: Expected value of stock if option is exercised
        - $K \\cdot e^{-rT} \\cdot N(d_2)$: Present value of strike price times probability of exercise
        
        **Put Option Intuition:**
        - $K \\cdot e^{-rT} \\cdot N(-d_2)$: Present value of strike price times probability of exercise
        - $S \\cdot N(-d_1)$: Expected value of stock if option is exercised
        
        **Key Insights:**
        - Higher volatility ($\\sigma$) → Higher option prices
        - Longer time ($T$) → Higher option prices (more uncertainty)
        - Higher interest rates ($r$) → Higher call prices, lower put prices
        """)
    
    # Step 8: Put-Call Parity
    with st.expander("8️⃣ Put-Call Parity", expanded=False):
        st.markdown("""
        An important relationship between call and put prices:
        """)
        st.latex(r"C - P = S - K \cdot e^{-rT}")
        
        # Calculate with current values
        try:
            call_price = black_scholes(S, K, T, r, sigma, "call")
            put_price = black_scholes(S, K, T, r, sigma, "put")
            left_side = call_price - put_price
            right_side = S - K * np.exp(-r * T)
            
            st.markdown(f"""
            **Verification with your inputs:**
            - Left side: $C - P = {call_price:.4f} - {put_price:.4f} = {left_side:.4f}$
            - Right side: $S - K \cdot e^{{-{r:.4f} \cdot {T:.4f}}} = {right_side:.4f}$
            - Difference: ${abs(left_side - right_side):.6f}$
            """)
        except:
            st.info("Enter valid parameters above to see the verification.")
    
    st.markdown("---")
    st.markdown("""
    **Congratulations!** You've now understood the mathematical foundation of the Black-Scholes model.
    This formula revolutionized options trading and remains one of the most important tools in quantitative finance.
    """)

if __name__ == "__main__":
    main()
