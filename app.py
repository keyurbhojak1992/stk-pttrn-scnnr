import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import mplfinance as mpf
import numpy as np
import cv2
import io
import os
import re
import json
import time  # <--- Added to fix Rate Limit Error

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(page_title="ProTrade: Master Scanner", layout="wide")

# Image Folders (Your Git Structure)
PATTERN_FOLDERS = {
    "Tight Base": "reference_images/tight_base",
    "Box Pattern": "reference_images/box_pattern",
    "VCP": "reference_images/vcp",
    "Cup & Handle": "reference_images/cup",
    "Flag & Pole": "reference_images/flag"
}

# --- 2. LOAD WATCHLISTS (FROM JSON) ---
@st.cache_data
def load_watchlists():
    if os.path.exists('watchlists.json'):
        with open('watchlists.json', 'r') as f:
            return json.load(f)
    # Fallback if file missing
    return {"Default": ["RELIANCE", "TCS", "INFY"]}

# --- 3. ULTIMATE DATA FETCHER (FIXED) ---
def get_valid_data(ticker):
    """
    Smart fetch: Tries Ticker -> Ticker.NS -> Ticker.BO -> US Ticker
    """
    ticker = str(ticker).strip().upper().replace(" ", "")
    
    # Handle prefixes
    if "NSE:" in ticker: ticker = ticker.replace("NSE:", "") + ".NS"
    elif "BSE:" in ticker: ticker = ticker.replace("BSE:", "") + ".BO"
    
    variations = []
    if "." in ticker:
        variations.append(ticker)
    else:
        variations.append(f"{ticker}.NS") # Priority 1: NSE
        variations.append(f"{ticker}.BO") # Priority 2: BSE
        variations.append(ticker)         # Priority 3: Global
    
    for symbol in variations:
        try:
            # FIX 1: Add delay to avoid Rate Limit Error
            time.sleep(0.5) 
            
            data = yf.download(symbol, period="1y", interval="1d", progress=False)
            
            # FIX 2: Flatten MultiIndex (Fixes AttributeError for pandas_ta)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            if not data.empty and len(data) > 50:
                return data, symbol
        except Exception:
            continue
    return None, None

# --- 4. STANDARD MATH LOGIC ENGINE ---
def analyze_pattern_math(df, pattern_name):
    """
    The 'Standard' Logic. Returns a Setup Dict if criteria met.
    """
    setup = {"Match": False, "Reason": ""}
    
    # Pre-calculate common indicators
    close = df['Close']
    high = df['High']
    low = df['Low']
    sma50 = df.ta.sma(length=50).iloc[-1]
    sma200 = df.ta.sma(length=200).iloc[-1]
    
    current_price = close.iloc[-1]

    # === LOGIC: FLAG AND POLE ===
    if pattern_name == "Flag & Pole":
        lookback_pole = 25
        start_price = close.iloc[-lookback_pole]
        peak_price = high.tail(lookback_pole).max()
        
        pole_move = (peak_price - start_price) / start_price
        
        if pole_move > 0.15:  # Pole exists (>15% gain)
            recent_consolidation = df.tail(10) 
            flag_low = recent_consolidation['Low'].min()
            flag_high = recent_consolidation['High'].max()
            
            retracement_limit = peak_price - (0.5 * (peak_price - start_price))
            
            if flag_low > retracement_limit:
                setup["Match"] = True
                setup["Entry"] = round(flag_high * 1.01, 2)
                setup["Stop"] = round(flag_low * 0.99, 2)
                setup["Target"] = round(setup["Entry"] + (peak_price - start_price), 2)
                setup["Reason"] = f"Pole +{round(pole_move*100)}%, Tight Flag"

    # === LOGIC: TIGHT BASE / BOX ===
    elif pattern_name == "Tight Base" or pattern_name == "Box Pattern":
        lookback = 20
        box_high = high.tail(lookback).max()
        box_low = low.tail(lookback).min()
        
        range_pct = (box_high - box_low) / box_low
        
        if range_pct < 0.10: 
            if current_price > sma50: 
                setup["Match"] = True
                setup["Entry"] = round(box_high * 1.005, 2)
                setup["Stop"] = round(box_low * 0.99, 2)
                setup["Target"] = round(box_high + (box_high - box_low) * 2, 2)
                setup["Reason"] = f"Tight {round(range_pct*100, 1)}% Range"

    # === LOGIC: VCP (Volatility Contraction) ===
    elif pattern_name == "VCP":
        range_big = high.tail(20).max() - low.tail(20).min()
        range_small = high.tail(10).max() - low.tail(10).min()
        
        if range_small < (range_big * 0.7): 
            if current_price > sma200:
                setup["Match"] = True
                pivot = high.tail(5).max()
                setup["Entry"] = round(pivot * 1.01, 2)
                setup["Stop"] = round(low.tail(5).min() * 0.99, 2)
                setup["Target"] = round(setup["Entry"] * 1.15, 2)
                setup["Reason"] = "Volatility Contracted > 30%"

    return setup

# --- 5. IMAGE MATCHING ENGINE ---
def analyze_pattern_visual(df, folder_path):
    buf = io.BytesIO()
    try:
        mpf.plot(df.tail(60), type='line', linecolor='black', axisoff=True, volume=False, savefig=buf)
        buf.seek(0)
        file_bytes = np.asarray(bytearray(buf.read()), dtype=np.uint8)
        live_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    except: return 0

    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(live_img, None)
    if des1 is None: return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    if not os.path.exists(folder_path): return 0
    valid_images = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg'))]
    if not valid_images: return 0

    scores = []
    for img_name in valid_images:
        path = os.path.join(folder_path, img_name)
        ref_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if ref_img is None: continue
        kp2, des2 = orb.detectAndCompute(ref_img, None)
        if des2 is None: continue
        matches = bf.match(des1, des2)
        if matches:
            scores.append(len(matches) / min(len(kp1), len(kp2)) * 100)

    return np.mean(scores) if scores else 0

# --- 6. MAIN UI ---
st.title("🛡️ ProTrade: Hybrid Scanner")
watchlists = load_watchlists()

with st.sidebar:
    st.header("1. Strategy")
    mode = st.radio("Scan Mode:", ["Standard Math (Logic)", "Visual Match (Images)"])
    
    if mode == "Standard Math (Logic)":
        pattern_choice = st.selectbox("Select Pattern:", ["Flag & Pole", "Tight Base", "VCP", "Box Pattern"])
    else:
        pattern_choice = st.selectbox("Select Image Folder:", list(PATTERN_FOLDERS.keys()))

    st.header("2. Input")
    input_type = st.radio("Source:", ["JSON Watchlist", "Paste Symbols", "CSV"])
    
    tickers = []
    if input_type == "JSON Watchlist":
        list_name = st.selectbox("Choose List:", list(watchlists.keys()))
        tickers = watchlists[list_name]
    elif input_type == "Paste Symbols":
        txt = st.text_area("Symbols:", "RELIANCE, JTEKTINDIA")
        if txt: tickers = [x.strip() for x in re.split(r'[,\n\s]+', txt) if x.strip()]
    elif input_type == "CSV":
        f = st.file_uploader("Upload CSV")
        if f: tickers = pd.read_csv(f).iloc[:,0].astype(str).tolist()

    run = st.button("RUN SCANNER", type="primary")

if run and tickers:
    st.write(f"### Scanning {len(tickers)} stocks for '{pattern_choice}'...")
    results = []
    bar = st.progress(0)
    
    for i, t in enumerate(tickers):
        data, valid_symbol = get_valid_data(t)
        if data is not None:
            # Global Safety 
            sma200 = data.ta.sma(length=200).iloc[-1] if len(data) > 200 else 0
            if data['Close'].iloc[-1] > sma200:
                
                match = False
                setup = {}
                
                if mode == "Standard Math (Logic)":
                    res = analyze_pattern_math(data, pattern_choice)
                    if res["Match"]:
                        match = True
                        setup = res
                        setup["Score"] = "Math Verified"
                else:
                    folder = PATTERN_FOLDERS.get(pattern_choice)
                    if folder:
                        score = analyze_pattern_visual(data, folder)
                        if score > 40:
                            match = True
                            p = data['Close'].iloc[-1]
                            setup = {
                                "Entry": round(p*1.01, 2), 
                                "Stop": round(p*0.95, 2), 
                                "Target": round(p*1.15, 2),
                                "Score": f"{round(score)}% Visual"
                            }

                if match:
                    results.append({
                        "Ticker": valid_symbol,
                        "Price": round(data['Close'].iloc[-1], 2),
                        "Type": setup["Score"],
                        "Entry": setup["Entry"],
                        "Stop": setup["Stop"],
                        "Target": setup["Target"]
                    })
                    st.session_state[f"data_{valid_symbol}"] = data
                    
        bar.progress((i+1)/len(tickers))
    
    if results:
        df_res = pd.DataFrame(results)
        st.success(f"Found {len(results)} matches!")
        st.dataframe(df_res)
        
        st.divider()
        sel = st.selectbox("Inspect Chart:", df_res['Ticker'].tolist())
        if sel:
            d = st.session_state[f"data_{sel}"]
            row = df_res[df_res['Ticker'] == sel].iloc[0]
            
            ap = [
                mpf.make_addplot([row['Entry']]*60, color='blue', linestyle='--'),
                mpf.make_addplot([row['Stop']]*60, color='red', linestyle='--')
            ]
            fig, ax = mpf.plot(d.tail(60), type='candle', style='yahoo', volume=True, addplot=ap, returnfig=True, title=sel)
            st.pyplot(fig)
    else:
        st.warning("No matches found.")
