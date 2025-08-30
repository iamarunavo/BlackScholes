import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Black-Scholes Model",
    page_icon="🚀",
    layout="wide"
)

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
    # Check matplotlib backend
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend for Streamlit
    except ImportError:
        st.error("Matplotlib is not installed. Please install it with: pip install matplotlib")
        return
    
    st.title("Black-Scholes Pricing Model")
    
    # Sidebar for inputs
    st.sidebar.header("Black-Scholes Model")
    
    # Creator info
    st.sidebar.write("**Created by:** Arunavo Chowdhury, Yaseen Choudhury")
    
    # Main input parameters
    st.sidebar.subheader("Input Parameters")
    S = st.sidebar.number_input("Current Asset Price", value=100.00, step=0.01)
    K = st.sidebar.number_input("Strike Price", value=100.00, step=0.01)
    T = st.sidebar.number_input("Time to Maturity (Years)", value=1.00, step=0.01)
    sigma = st.sidebar.number_input("Volatility (σ)", value=0.20, step=0.01)
    r = st.sidebar.number_input("Risk-Free Interest Rate", value=0.05, step=0.001)
    
    # Heatmap parameters
    st.sidebar.subheader("Heatmap Parameters")
    min_spot = st.sidebar.number_input("Min Spot Price", value=80.00, step=0.01)
    max_spot = st.sidebar.number_input("Max Spot Price", value=120.00, step=0.01)
    min_vol = st.sidebar.number_input("Min Volatility for Heatmap", value=0.10, min_value=0.01, max_value=1.00, step=0.01)
    max_vol = st.sidebar.number_input("Max Volatility for Heatmap", value=0.30, min_value=0.01, max_value=1.00, step=0.01)
    
    # Create ranges for heatmap
    spot_range = np.linspace(min_spot, max_spot, 10)
    vol_range = np.linspace(min_vol, max_vol, 10)
    
    # Display current parameters
    st.subheader("Current Parameters")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.write(f"**Current Asset Price:** {S:.4f}")
    with col2:
        st.write(f"**Strike Price:** {K:.4f}")
    with col3:
        st.write(f"**Time to Maturity (Years):** {T:.4f}")
    with col4:
        st.write(f"**Volatility (σ):** {sigma:.4f}")
    with col5:
        st.write(f"**Risk-Free Interest Rate:** {r:.4f}")
    
    # Calculate option prices
    try:
        call_price = black_scholes(S, K, T, r, sigma, "call")
        put_price = black_scholes(S, K, T, r, sigma, "put")
        
        # Display option values in prominent boxes
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div style="background-color: #00ff88; padding: 20px; border-radius: 10px; text-align: center;">
                <h2 style="color: #000000; margin: 0;">CALL Value</h2>
                <h1 style="color: #000000; margin: 10px 0;">${call_price:.2f}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background-color: #ff69b4; padding: 20px; border-radius: 10px; text-align: center;">
                <h2 style="color: #000000; margin: 0;">PUT Value</h2>
                <h1 style="color: #000000; margin: 10px 0;">${put_price:.2f}</h1>
            </div>
            """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error calculating option price: {str(e)}")
    
    # Heatmap section
    st.header("Options Price - Interactive Heatmap")
    st.write("Explore how option prices fluctuate with varying 'Spot Prices and Volatility' levels using interactive heatmap parameters, all while maintaining a constant 'Strike Price'.")
    
    # Create heatmaps using seaborn
    try:
        # Prepare model data for heatmap function
        bs_model = {'T': T, 'r': r}
        
        # Create side-by-side heatmaps
        col1, col2 = st.columns([1,1], gap="small")
        
        with col1:
            st.subheader("Call Price Heatmap")
            heatmap_fig_call, _ = plot_heatmap(bs_model, spot_range, vol_range, K)
            st.pyplot(heatmap_fig_call)
        
        with col2:
            st.subheader("Put Price Heatmap")
            _, heatmap_fig_put = plot_heatmap(bs_model, spot_range, vol_range, K)
            st.pyplot(heatmap_fig_put)
        
    except Exception as e:
        st.error(f"Error creating heatmaps: {str(e)}")
        st.write("Please check that matplotlib and seaborn are properly installed and try again.")
        return
    
    # Display current option prices
    st.subheader("Current Option Prices")
    st.write(f"Call Option Price: **${black_scholes(S, K, T, r, sigma, 'call'):.2f}**")
    st.write(f"Put Option Price: **${black_scholes(S, K, T, r, sigma, 'put'):.2f}**")
    
    # Calculate prices for summary statistics
    call_prices = np.zeros((len(vol_range), len(spot_range)))
    put_prices = np.zeros((len(vol_range), len(spot_range)))
    
    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            try:
                call_prices[i, j] = black_scholes(spot, K, T, r, vol, "call")
                put_prices[i, j] = black_scholes(spot, K, T, r, vol, "put")
            except:
                call_prices[i, j] = np.nan
                put_prices[i, j] = np.nan
    
    # Display summary statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Max Call Price", f"${np.nanmax(call_prices):.4f}")
        st.metric("Min Call Price", f"${np.nanmin(call_prices):.4f}")
    
    with col2:
        st.metric("Max Put Price", f"${np.nanmax(put_prices):.4f}")
        st.metric("Min Put Price", f"${np.nanmin(put_prices):.4f}")
    
    with col3:
        st.metric("Call-Put Spread", f"${np.nanmax(call_prices - put_prices):.4f}")
        st.metric("Price Range", f"${np.nanmax(call_prices) - np.nanmin(call_prices):.4f}")
    
    # Additional analysis section
    st.header("Black-Scholes Formula")
    st.latex(r'''
    \text{Call Price} = S \cdot N(d_1) - K \cdot e^{-rT} \cdot N(d_2)
    ''')
    st.latex(r'''
    \text{Put Price} = K \cdot e^{-rT} \cdot N(-d_2) - S \cdot N(-d_1)
    ''')
    st.latex(r'''
    d_1 = \frac{\ln(S/K) + (r + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}
    ''')
    st.latex(r'''
    d_2 = d_1 - \sigma\sqrt{T}
    ''')
    
    st.write("Where:")
    st.write("- S = Current stock price")
    st.write("- K = Strike price")
    st.write("- T = Time to maturity")
    st.write("- r = Risk-free interest rate")
    st.write("- σ = Volatility")
    st.write("- N(·) = Cumulative normal distribution function")

if __name__ == "__main__":
    main()
