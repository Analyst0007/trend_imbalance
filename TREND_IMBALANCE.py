import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Financial Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">📈 Financial Analysis Dashboard</h1>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("📊 Configuration Panel")

# Stock Symbol Input
ticker = st.sidebar.text_input(
    "Stock Symbol", 
    value="^NSEI",
    help="Enter stock symbol (e.g., AAPL, ^NSEI, TSLA)"
)

# Date Range Selection
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        value=datetime.now() - timedelta(days=365),
        max_value=datetime.now()
    )
with col2:
    end_date = st.date_input(
        "End Date",
        value=datetime.now(),
        max_value=datetime.now()
    )

# Data Frequency Selection
frequency_options = {
    "1 Day": "1d",
    "1 Week": "1wk", 
    "1 Month": "1mo",
    "3 Months": "3mo",
    "1 min":"1m",
    "5 min":"5m",
    "15 min":"15m",
    
    
    
}
frequency = st.sidebar.selectbox(
    "Data Frequency",
    options=list(frequency_options.keys()),
    index=0,
    help="Choose the frequency for data download"
)

# Window Selection Options
st.sidebar.subheader("🔧 Analysis Parameters")

window_size = st.sidebar.slider(
    "Moving Window Size",
    min_value=5,
    max_value=50,
    value=20,
    help="Window size for support/resistance calculation"
)

min_distance = st.sidebar.slider(
    "Minimum Distance Filter",
    min_value=0.1,
    max_value=2.0,
    value=0.5,
    step=0.1,
    help="Minimum distance between support/resistance levels"
)

price_range_pct = st.sidebar.slider(
    "Price Range Percentage",
    min_value=0.001,
    max_value=0.01,
    value=0.002,
    step=0.001,
    format="%.3f",
    help="Price range percentage for signal generation"
)

transaction_fee = st.sidebar.slider(
    "Transaction Fee Rate",
    min_value=0.0,
    max_value=0.01,
    value=0.002,
    step=0.001,
    format="%.3f",
    help="Transaction fee rate for backtesting"
)

initial_cash = st.sidebar.number_input(
    "Initial Cash",
    min_value=1000,
    max_value=1000000,
    value=10000,
    step=1000,
    help="Initial cash for backtesting"
)

# Functions
@st.cache_data
def get_clean_financial_data(ticker, start_date, end_date, interval):
    """Fetch and clean stock data"""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        if data.empty:
            return None
        
        # Handle multi-level columns
        if len(data.columns.levels) > 1:
            data.columns = data.columns.get_level_values(0)
        
        data = data.ffill()
        data.index = data.index.tz_localize(None)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def calculate_order_flow(data):
    """Calculate order flow metrics"""
    data = data.copy()
    data['Price Change'] = data['Close'].diff()
    data['Buy Volume'] = data['Volume'].where(data['Price Change'] > 0, 0)
    data['Sell Volume'] = data['Volume'].where(data['Price Change'] < 0, 0)
    data['Accumulated Buy Volume'] = data['Buy Volume'].cumsum()
    data['Accumulated Sell Volume'] = data['Sell Volume'].cumsum()
    data['Order Flow Imbalance'] = data['Accumulated Buy Volume'] - data['Accumulated Sell Volume']
    return data

def find_support_resistance(data, window, min_distance):
    """Find support and resistance levels"""
    data = data.copy()
    data['Support'] = data['Close'].rolling(window).min()
    data['Resistance'] = data['Close'].rolling(window).max()
    
    support_levels = data['Support'].dropna().drop_duplicates().tolist()
    resistance_levels = data['Resistance'].dropna().drop_duplicates().tolist()

    def filter_levels(levels, min_distance):
        filtered = []
        for level in sorted(set(levels)):
            if not filtered or abs(level - filtered[-1]) >= min_distance:
                filtered.append(level)
        return filtered

    support_levels = filter_levels(support_levels, min_distance)
    resistance_levels = filter_levels(resistance_levels, min_distance)

    if support_levels and resistance_levels:
        max_support = max(support_levels)
        min_resistance = min(resistance_levels)
        support_levels = [lvl for lvl in support_levels if lvl < min_resistance]
        resistance_levels = [lvl for lvl in resistance_levels if lvl > max_support]

    return support_levels, resistance_levels

def generate_signals(data, support_levels, resistance_levels, price_range_pct):
    """Generate buy/sell signals"""
    data = data.copy()
    data['Buy Signal'] = False
    data['Sell Signal'] = False

    for i in range(1, len(data)):
        close_price = data['Close'].iloc[i]
        imbalance = data['Order Flow Imbalance'].iloc[i]

        for s in support_levels:
            if s * (1 - price_range_pct) <= close_price <= s * (1 + price_range_pct) and imbalance < 0:
                data.at[data.index[i], 'Buy Signal'] = True

        for r in resistance_levels:
            if r * (1 - price_range_pct) <= close_price <= r * (1 + price_range_pct) and imbalance > 0:
                data.at[data.index[i], 'Sell Signal'] = True

    return data

def backtest_strategy(data, initial_cash, transaction_fee_rate):
    """Perform backtesting with detailed metrics"""
    cash = initial_cash
    position = 0
    portfolio_values = []
    transaction_log = []
    wins = 0
    trades = 0

    for i in range(len(data) - 1):
        today = data.index[i]
        next_day = data.index[i + 1]
        next_open = data['Open'].iloc[i + 1]
        close_today = data['Close'].iloc[i]

        current_portfolio_value = cash + position * close_today

        # BUY
        if data['Buy Signal'].iloc[i] and position == 0:
            shares_to_buy = (cash * (1 - transaction_fee_rate)) / next_open
            cost = shares_to_buy * next_open
            transaction_log.append({
                'Date': next_day.strftime('%Y-%m-%d'),
                'Action': 'BUY',
                'Price': round(next_open, 2),
                'Shares': round(shares_to_buy, 2),
                'Cash Before': round(cash, 2),
                'Cash After': round(0, 2),
                'Portfolio Before': round(current_portfolio_value, 2),
                'Portfolio After': round(shares_to_buy * next_open, 2)
            })
            position = shares_to_buy
            cash = 0

        # SELL
        elif data['Sell Signal'].iloc[i] and position > 0:
            proceeds = position * next_open * (1 - transaction_fee_rate)
            new_value = proceeds
            if new_value > current_portfolio_value:
                wins += 1
            trades += 1
            transaction_log.append({
                'Date': next_day.strftime('%Y-%m-%d'),
                'Action': 'SELL',
                'Price': round(next_open, 2),
                'Shares': round(position, 2),
                'Cash Before': round(cash, 2),
                'Cash After': round(proceeds, 2),
                'Portfolio Before': round(current_portfolio_value, 2),
                'Portfolio After': round(new_value, 2)
            })
            cash = proceeds
            position = 0

        portfolio_values.append(current_portfolio_value)

    # Final value
    final_value = cash + position * data['Close'].iloc[-1]
    portfolio_values.append(final_value)
    
    return portfolio_values, transaction_log, wins, trades, final_value

def create_interactive_charts(data, support_levels, resistance_levels, portfolio_values):
    """Create interactive Plotly charts"""
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=['Price with Signals', 'Order Flow Analysis', 'Portfolio Performance'],
        vertical_spacing=0.08,
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Price chart with signals
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Close'], name='Close Price', line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    # Buy signals
    buy_signals = data[data['Buy Signal']]
    if not buy_signals.empty:
        fig.add_trace(
            go.Scatter(x=buy_signals.index, y=buy_signals['Close'], 
                      mode='markers', name='Buy Signal',
                      marker=dict(symbol='triangle-up', size=10, color='green')),
            row=1, col=1
        )
    
    # Sell signals
    sell_signals = data[data['Sell Signal']]
    if not sell_signals.empty:
        fig.add_trace(
            go.Scatter(x=sell_signals.index, y=sell_signals['Close'],
                      mode='markers', name='Sell Signal',
                      marker=dict(symbol='triangle-down', size=10, color='red')),
            row=1, col=1
        )
    
    # Support and resistance levels
    for level in support_levels:
        fig.add_hline(y=level, line_dash="dash", line_color="green", opacity=0.3, row=1, col=1)
    for level in resistance_levels:
        fig.add_hline(y=level, line_dash="dash", line_color="red", opacity=0.3, row=1, col=1)
    
    # Order flow chart
    daily_imbalance = data['Order Flow Imbalance'].diff().fillna(0)
    colors = ['green' if x > 0 else 'red' for x in daily_imbalance]
    
    fig.add_trace(
        go.Bar(x=data.index, y=daily_imbalance, name='Daily Imbalance',
               marker_color=colors, opacity=0.6),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Order Flow Imbalance'], 
                  name='Cumulative Imbalance', line=dict(color='black', width=1.5)),
        row=2, col=1
    )
    
    # Portfolio performance
    portfolio_dates = data.index[:len(portfolio_values)]
    fig.add_trace(
        go.Scatter(x=portfolio_dates, y=portfolio_values,
                  name='Portfolio Value', line=dict(color='purple', width=2)),
        row=3, col=1
    )
    
    fig.update_layout(height=800, showlegend=True, title_text=f"{ticker} Financial Analysis")
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Order Flow", row=2, col=1)
    fig.update_yaxes(title_text="Portfolio Value", row=3, col=1)
    
    return fig

# Main Analysis
if st.sidebar.button("🚀 Run Analysis", type="primary"):
    with st.spinner("Fetching and analyzing data..."):
        # Fetch data
        data = get_clean_financial_data(
            ticker, 
            start_date, 
            end_date, 
            frequency_options[frequency]
        )
        
        if data is None or data.empty or len(data) < 2:
            st.error("❌ Insufficient data for analysis. Please check your inputs.")
        else:
            # Calculate order flow
            data = calculate_order_flow(data)
            
            # Find support and resistance
            support_levels, resistance_levels = find_support_resistance(
                data, window_size, min_distance
            )
            
            # Generate signals
            data = generate_signals(data, support_levels, resistance_levels, price_range_pct)
            
            # Backtest
            portfolio_values, transaction_log, wins, trades, final_value = backtest_strategy(
                data, initial_cash, transaction_fee
            )
            
            # Calculate metrics
            total_return_pct = ((final_value - initial_cash) / initial_cash) * 100
            win_rate = (wins / trades) * 100 if trades > 0 else 0
            
            # Max drawdown
            running_max = np.maximum.accumulate(portfolio_values)
            drawdowns = (portfolio_values - running_max) / running_max
            max_drawdown = drawdowns.min()
            
            # Display Results
            st.success("✅ Analysis completed successfully!")
            
            # Performance Metrics
            st.subheader("📊 Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Initial Value",
                    value=f"${initial_cash:,.2f}"
                )
            
            with col2:
                st.metric(
                    label="Final Value",
                    value=f"${final_value:,.2f}",
                    delta=f"${final_value - initial_cash:,.2f}"
                )
            
            with col3:
                st.metric(
                    label="Total Return",
                    value=f"{total_return_pct:.2f}%"
                )
            
            with col4:
                st.metric(
                    label="Win Rate",
                    value=f"{win_rate:.2f}%"
                )
            
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                st.metric(
                    label="Total Trades",
                    value=f"{trades}"
                )
            
            with col6:
                st.metric(
                    label="Winning Trades",
                    value=f"{wins}"
                )
            
            with col7:
                st.metric(
                    label="Max Drawdown",
                    value=f"{max_drawdown:.2%}"
                )
            
            with col8:
                st.metric(
                    label="Support Levels",
                    value=f"{len(support_levels)}"
                )
            
            # Interactive Charts
            st.subheader("📈 Interactive Analysis Charts")
            fig = create_interactive_charts(data, support_levels, resistance_levels, portfolio_values)
            st.plotly_chart(fig, use_container_width=True)
            
            # Transaction Log
            if transaction_log:
                st.subheader("📋 Transaction Log")
                log_df = pd.DataFrame(transaction_log)
                st.dataframe(log_df, use_container_width=True)
                
                # Download button for transaction log
                csv = log_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Transaction Log",
                    data=csv,
                    file_name=f"{ticker}_transaction_log.csv",
                    mime="text/csv"
                )
            else:
                st.info("No transactions were executed during the analysis period.")
            
            # Additional Information
            with st.expander("ℹ️ Analysis Details"):
                st.write(f"**Data Points:** {len(data)}")
                st.write(f"**Support Levels:** {support_levels}")
                st.write(f"**Resistance Levels:** {resistance_levels}")
                st.write(f"**Buy Signals:** {data['Buy Signal'].sum()}")
                st.write(f"**Sell Signals:** {data['Sell Signal'].sum()}")

else:
    # Show instructions when no analysis is running
    st.info("👆 Configure your analysis parameters in the sidebar and click 'Run Analysis' to begin.")
    
    st.markdown("""
    ### 🔍 How to Use This Dashboard:
    
    1. **Enter Stock Symbol**: Input the ticker symbol (e.g., AAPL, ^NSEI, TSLA)
    2. **Select Date Range**: Choose start and end dates for analysis
    3. **Choose Data Frequency**: Select from daily, weekly, or monthly data
    4. **Adjust Parameters**: Fine-tune analysis settings in the sidebar
    5. **Run Analysis**: Click the button to start the analysis
    
    ### 📈 Features:
    - **Order Flow Analysis**: Tracks buying and selling pressure
    - **Support/Resistance Detection**: Automatically identifies key levels
    - **Signal Generation**: Creates buy/sell signals based on technical analysis
    - **Backtesting**: Tests strategy performance with transaction costs
    - **Interactive Charts**: Zoom, pan, and explore your data
    - **Transaction Log**: Detailed record of all trades
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Financial Analysis Dashboard | Built with Streamlit 📊"
    "</div>", 
    unsafe_allow_html=True
)