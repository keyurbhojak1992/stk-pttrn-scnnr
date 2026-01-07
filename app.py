import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import mplfinance as mpf
import numpy as np
import cv2
import os

# --- 1. CONFIGURATION & PAGE SETUP ---
st.set_page_config(page_title="ProTrade Hybrid Scanner", layout="wide")

# Standard "Math" Variables for patterns (The "Database" of logic)
PATTERN_PARAMS = {
    "VCP": {"lookback": 20, "contraction_threshold": 0.75, "risk_reward": 3},
    "Cup & Handle": {"min_cup_days": 20, "handle_max_drop": 0.33, "risk_reward": 2.5},
    "Bull Flag": {"pole_move_min": 0.15, "flag_consolidation_max": 0.25, "risk_reward": 2},
    "Tight Base / Box": {"box_range_percent": 0.10, "min_days": 15, "risk_reward": 3}
}

# --- 2. SMART SYMBOL HANDLER ---
def get_valid_data(ticker):
    """
    Tries to fetch data for ticker. 
    1. Tries Ticker as is.
    2. If empty, tries appending .NS (NSE India).
    3. Returns Data and the 'Working Ticker' name.
    """
    # Attempt 1: As provided
    data = yf.download(ticker, period="1y", progress=False)
    if not data.empty:
        return data, ticker
    
    # Attempt 2: Append .NS (for Indian Stocks)
    if not str(ticker).endswith(".NS"):
        try_ticker = f"{ticker}.NS"
        data = yf.download(try_ticker, period="1y", progress=False)
        if not data.empty:
            return data, try_ticker
            
    return None, None

# --- 3. SAFETY & TREND FILTER (Global) ---
def check_global_safety(df):
    if df.empty or len(df) < 200: return False
    
    current_close = df['Close'].iloc[-1]
    sma50 = df.ta.sma(length=50).iloc[-1]
    sma200 = df.ta.sma(length=200).iloc[-1]
    rsi = df.ta.rsi(length=14).iloc[-1]
    vol_avg = df['Volume'].rolling(20).mean().iloc[-1]

    # 1. Trend: Price > 200 SMA (Long term uptrend)
    if current_close < sma200: return False
    
    # 2. No Pump/Dump: RSI not extreme (> 85 is dangerous)
    if rsi > 85: return False
    
    # 3. Minimum Liquidity (Avoid dead stocks)
    if vol_avg < 50000: return False

    return True

# --- 4. PATTERN RECOGNITION ENGINE (MATH BASE) ---
def analyze_pattern_math(ticker, df, pattern_type):
    """
    Checks for patterns using pure mathematical rules (Standard Database).
    Returns a dictionary with Trade Plan if match found.
    """
    last_close = df['Close'].iloc[-1]
    setup = {"Ticker": ticker, "Pattern": pattern_type, "Match": False}

    # === PATTERN: VCP (Volatility Contraction) ===
    if pattern_type == "VCP":
        # Check if volatility is shrinking
        range_recent = df['High'].tail(10).max() - df['Low'].tail(10).min()
        range_prev = df['High'].tail(20).max() - df['Low'].tail(20).min()
        
        if range_recent < (range_prev * PATTERN_PARAMS["VCP"]["contraction_threshold"]):
            pivot = df['High'].tail(5).max()
            stop = df['Low'].tail(5).min()
            
            setup.update({
                "Match": True,
                "Entry": round(pivot * 1.005, 2),
                "Stop_Loss": round(stop * 0.995, 2),
                "Target": round(pivot + ((pivot - stop) * 3), 2),
                "Trailing_SL": "EMA 10"
            })

    # === PATTERN: BULL FLAG ===
    elif pattern_type == "Bull Flag":
        # Pole: 15% move up in 20 days
        move_20d = (df['Close'].iloc[-1] - df['Close'].iloc[-20]) / df['Close'].iloc[-20]
        
        if move_20d > PATTERN_PARAMS["Bull Flag"]["pole_move_min"]:
            # Flag: Recent 5 days consolidation
            flag_high = df['High'].tail(5).max()
            flag_low = df['Low'].tail(5).min()
            
            # Check flag isn't too deep
            if flag_low > (df['Close'].iloc[-20] * 0.85): 
                setup.update({
                    "Match": True,
                    "Entry": round(flag_high * 1.01, 2),
                    "Stop_Loss": round(flag_low * 0.99, 2),
                    "Target": round(flag_high * 1.20, 2), # Projected move
                    "Trailing_SL": "Low of Prev 3 Candles"
                })

    # === PATTERN: TIGHT BASE / BOX ===
    elif pattern_type == "Tight Base / Box":
        # Price range < 10% for last 15 days
        box_high = df['High'].tail(15).max()
        box_low = df['Low'].tail(15).min()
        range_pct = (box_high - box_low) / box_low
        
        if range_pct < PATTERN_PARAMS["Tight Base / Box"]["box_range_percent"]:
            setup.update({
                "Match": True,
                "Entry": round(box_high * 1.01, 2),
                "Stop_Loss": round(box_low * 0.99, 2),
                "Target": round(box_high + (box_high - box_low) * 2, 2),
                "Trailing_SL": "SMA 20"
            })

    # === PATTERN: CUP AND HANDLE ===
    elif pattern_type == "Cup & Handle":
        # Simplified: Price is near 52 week high but slightly below
        year_high = df['High'].tail(250).max()
        if last_close > (year_high * 0.85) and last_close < (year_high * 0.98):
             setup.update({
                "Match": True,
                "Entry": round(year_high * 1.01, 2),
                "Stop_Loss": round(last_close * 0.95, 2),
                "Target": round(year_high * 1.25, 2),
                "Trailing_SL": "SMA 50"
            })

    return setup

# --- 5. IMAGE MATCHING PLACEHOLDER ---
def analyze_pattern_visual(ticker, df, uploaded_ref_images):
    """
    (Advanced) Compares chart image to uploaded reference images.
    For this demo, we simulate a match if the Math logic is also true.
    """
    # In full production: 
    # 1. Generate chart image from 'df'
    # 2. Use OpenCV to compare with 'uploaded_ref_images'
    # 3. Return Match Score
    return analyze_pattern_math(ticker, df, "VCP") # Fallback to math for demo

# --- 6. CHART PLOTTER ---
def plot_chart(ticker, data, setup):
    # Professional Style
    mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', wick='black', edge='inherit')
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', y_on_right=True)
    
    # Add Entry/Stop lines
    addplots = [
        mpf.make_addplot([setup['Entry']]*len(data.tail(100)), color='blue', linestyle='--'),
        mpf.make_addplot([setup['Stop_Loss']]*len(data.tail(100)), color='red', linestyle='--'),
        mpf.make_addplot(data['Close'].tail(100).rolling(50).mean(), color='orange') # 50 SMA
    ]
    
    fig, ax = mpf.plot(
        data.tail(100), type='candle', style=s, volume=True,
        addplot=addplots,
        title=f"\n{ticker} - {setup['Pattern']} Setup",
        returnfig=True
    )
    st.pyplot(fig)

# --- 7. MAIN DASHBOARD ---
st.title("🚀 ProTrade: Hybrid Pattern Scanner")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # 1. METHOD SELECTION
    scan_method = st.radio("Scanning Method:", ["Standard Math (Fast)", "Image Similarity (Advanced)"])
    
    # 2. PATTERN SELECTION
    if scan_method == "Standard Math (Fast)":
        target_pattern = st.selectbox("Select Pattern to Find:", list(PATTERN_PARAMS.keys()))
        ref_images = None
    else:
        target_pattern = st.text_input("Name this Pattern:", "Custom Setup")
        ref_images = st.file_uploader("Upload Reference Chart Images", accept_multiple_files=True, type=['png', 'jpg'])

    # 3. DATA INPUT
    st.divider()
    stock_file = st.file_uploader("Upload Stock List (CSV)", type=['csv'])
    
    run_btn = st.button("RUN SCANNER", type="primary")

# --- MAIN LOGIC ---
if run_btn and stock_file:
    st.write(f"### 🔍 Scanning for: {target_pattern}...")
    
    # Load List
    df_stocks = pd.read_csv(stock_file)
    tickers = df_stocks.iloc[:, 0].astype(str).tolist()
    
    results = []
    progress = st.progress(0)
    status_text = st.empty()
    
    for i, t in enumerate(tickers):
        status_text.text(f"Checking {t}...")
        
        # 1. SMART DATA FETCH
        data, valid_ticker = get_valid_data(t)
        
        if data is not None and len(data) > 200:
            # 2. GLOBAL SAFETY CHECK
            if check_global_safety(data):
                
                # 3. PATTERN CHECK (Dual Mode)
                if scan_method == "Standard Math (Fast)":
                    setup = analyze_pattern_math(valid_ticker, data, target_pattern)
                else:
                    # Image Mode (Requires Ref Images)
                    if ref_images:
                        setup = analyze_pattern_visual(valid_ticker, data, ref_images)
                    else:
                        st.error("Please upload reference images for Visual Mode.")
                        break
                
                # 4. SAVE IF MATCH
                if setup["Match"]:
                    results.append(setup)
                    # Save data for chart view
                    st.session_state[f"data_{valid_ticker}"] = data
        
        progress.progress((i + 1) / len(tickers))
    
    status_text.text("Scan Complete!")
    
    # --- DISPLAY RESULTS ---
    if results:
        st.success(f"✅ Found {len(results)} Opportunities")
        results_df = pd.DataFrame(results).drop(columns=["Match"])
        
        # Display Interactive Table
        st.dataframe(results_df.style.format({"Entry": "{:.2f}", "Stop_Loss": "{:.2f}", "Target": "{:.2f}"}))
        
        st.divider()
        
        # --- CHART VIEWER ---
        st.header("📊 Pattern Visualizer")
        selected_stock = st.selectbox("Select Stock to Inspect:", [r['Ticker'] for r in results])
        
        if selected_stock:
            # Retrieve saved data and setup info
            stock_data = st.session_state[f"data_{selected_stock}"]
            stock_setup = next(item for item in results if item["Ticker"] == selected_stock)
            
            # Show Chart with Levels
            plot_chart(selected_stock, stock_data, stock_setup)
            
            # Show Trading Plan Details
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("ENTRY", stock_setup['Entry'])
            c2.metric("STOP LOSS", stock_setup['Stop_Loss'])
            c3.metric("TARGET", stock_setup['Target'])
            c4.info(f"**Trailing SL:** {stock_setup['Trailing_SL']}")
            
    else:
        st.warning("No stocks matched your criteria today.")

elif run_btn and not stock_file:
    st.error("Please upload a CSV file first.")