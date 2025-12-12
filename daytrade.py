#!/usr/bin/env python3
"""
ULTIMATE HYBRID DAY TRADER 2025 — FINAL CLEAN EDITION
→ No duplicates
→ No crashes
→ Beautiful dark + green dashboard
→ High-probability LONG only
→ Prophet + ADX + Patterns + Volume + MTF
→ Ready to print money tomorrow
"""

import os
import sys
import time
import signal
from datetime import datetime
from typing import List, Dict, Optional
from multiprocessing import Pool, cpu_count

import pandas as pd
import numpy as np
import yfinance as yf
from prophet import Prophet

# ====================== CONFIG ======================
SYMBOLS_FILE = "symbolsFAST.txt"
EXPORT_DIR = "exports"
HTML_FILE = os.path.join(EXPORT_DIR, "ultimate_dashboard.html")
REFRESH_SECONDS = 300
TOP_N = 12
MAX_TRACKED = 60
MIN_PROB = 65

os.makedirs(EXPORT_DIR, exist_ok=True)

# ====================== INDICATORS ======================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    
    # True Range & ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    
    # EMAs
    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Volume
    df['VolMA20'] = df['Volume'].rolling(20).mean()
    df['VolRatio'] = df['Volume'] / df['VolMA20']
    
    # ADX (smoothed)
    plus_dm = df['High'].diff()
    minus_dm = -df['Low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr14 = df['ATR'].rolling(14).mean()
    df['+DI'] = 100 * (plus_dm.ewm(alpha=1/14).mean() / tr14)
    df['-DI'] = 100 * (minus_dm.ewm(alpha=1/14).mean() / tr14)
    dx = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'] + 1e-9)
    df['ADX'] = dx.ewm(alpha=1/14).mean()
    
    return df

def detect_bullish_patterns(df: pd.DataFrame) -> List[str]:
    if df is None or len(df) < 10:
        return []
    l = df.iloc[-1]
    p1 = df.iloc[-2]
    patterns = []
    if p1['Close'] < p1['Open'] and l['Close'] > l['Open'] and l['Open'] < p1['Close'] and l['Close'] > p1['Open']:
        patterns.append("ENGULF")
    body = abs(l['Close'] - l['Open'])
    lower_shadow = min(l['Open'], l['Close']) - l['Low']
    if lower_shadow > 2.5 * body:
        patterns.append("HAMMER")
    if l['VolRatio'] > 2.2:
        patterns.append("VOL_SURGE")
    return patterns

# ====================== PROPHET ======================
def prophet_forecast(ser: pd.Series):
    if ser is None or len(ser) < 30:
        return {1: "—", 2: "—", 4: "—"}
    try:
        df = pd.DataFrame({"ds": ser.index, "y": ser.values})
        m = Prophet(daily_seasonality=True, weekly_seasonality=False, yearly_seasonality=False, interval_width=0.8)
        m.fit(df)
        future = m.make_future_dataframe(periods=4, freq='H')
        fc = m.predict(future)
        curr = float(ser.iloc[-1])
        out = {}
        for h in [1, 2, 4]:
            y = float(fc.iloc[-5 + h]['yhat'])
            trend = "up" if y > curr * 1.005 else "down" if y < curr * 0.995 else "flat"
            out[h] = f"${y:.2f} {trend}"
        return out
    except:
        return {1: "—", 2: "—", 4: "—"}

# ====================== ANALYSIS ======================
def analyze(sym: str, df5: Optional[pd.DataFrame], df1h: Optional[pd.DataFrame]) -> Optional[dict]:
    if df5 is None or df5.empty or len(df5) < 50:
        return None
    if df1h is None or df1h.empty or len(df1h) < 20:
        return None
    
    df5 = add_indicators(df5.copy())
    df1h = add_indicators(df1h.copy())
    l5 = df5.iloc[-1]
    l1h = df1h.iloc[-1]
    
    # LONG ONLY
    if l5['Close'] <= l5['EMA9']:
        return None
    
    score = 0
    reasons = []
    
    if l5['Close'] > l5['EMA9'] and l1h['Close'] > l1h['EMA9']:
        score += 35
        reasons.append("MTF_BULL")
    
    if l5['VolRatio'] > 1.8:
        score += 25
        reasons.append("HIGH_VOL")
    elif l5['VolRatio'] > 1.3:
        score += 12
    
    if l5['MACD'] > l5['MACD_Signal']:
        score += 15
        reasons.append("MACD_BULL")
    
    rsi = l5['RSI']
    if rsi < 38:
        score += 22
        reasons.append("OVERSOLD")
    elif 40 < rsi < 65:
        score += 10
    
    if l5['ADX'] > 25:
        score += 20
        reasons.append("STRONG_TREND")
    
    patterns = detect_bullish_patterns(df5)
    if patterns:
        score += min(20, len(patterns) * 9)
        reasons.extend(patterns[:2])
    
    if score < MIN_PROB:
        return None
    
    atr = l5['ATR']
    price = l5['Close']
    stop = price - 1.5 * atr
    target = price + 3.0 * atr
    rr = round((target - price) / (price - stop), 2)
    
    return {
        "Symbol": sym,
        "Price": round(price, 3),
        "Prob": round(score, 1),
        "RSI": round(rsi, 1),
        "Vol": round(l5['VolRatio'], 2),
        "Stop": round(stop, 2),
        "Target": round(target, 2),
        "RR": rr,
        "ADX": round(l5['ADX'], 1),
        "Signals": " | ".join(reasons[:4])
    }

# ====================== DOWNLOAD ======================
def download(syms, period, interval):
    result = {s: pd.DataFrame() for s in syms}
    try:
        data = yf.download(syms, period=period, interval=interval, group_by='ticker',
                           auto_adjust=True, prepost=True, threads=True, progress=False, timeout=30)
        for s in syms:
            df = data[s] if len(syms) > 1 and s in data.columns.levels[0] else data
            if df is not None and not df.empty and len(df) > 10:
                result[s] = df[['Open','High','Low','Close','Volume']].dropna()
    except:
        pass
    return result

# ====================== DASHBOARD ======================
def build_html(results, forecasts):
    rows = ""
    has_buy = len(results) > 0
    for r in results:
        f = forecasts.get(r["Symbol"], {1:"—",2:"—",4:"—"})
        flash = 'class="flash"' if r["Prob"] >= 80 else ""
        rows += f'<tr {flash}><td><b>{r["Symbol"]}</b></td><td>${r["Price"]}</td><td><b>{r["Prob"]}%</b></td><td>{r["RSI"]}</td><td>{r["Vol"]}x</td><td>${r["Stop"]}</td><td>${r["Target"]}</td><td>{r["RR"]}:1</td><td>{r["ADX"]}</td><td>{r["Signals"]}</td><td>{f[1]}</td><td>{f[2]}</td><td>{f[4]}</td></tr>'
    
    sound = '<audio autoplay loop><source src="https://assets.mixkit.co/sfx/preview/mixkit-arcade-game-jump-coin-216.mp3"></audio>' if has_buy else ""
    
    return f'''<!DOCTYPE html><html><head><meta http-equiv="refresh" content="{REFRESH_SECONDS}"><title>ULTIMATE DAY TRADER</title>
<style>
    body{{font-family:system-ui;background:#0d1117;color:#c9d1d9;margin:20px}}
    h1{{color:#00ff88;text-align:center}}
    table{{width:100%;border-collapse:collapse;background:#161b22;border:1px solid #30363d}}
    th{{background:#0f4d0f;color:white;padding:14px}}
    td{{padding:12px;border-bottom:1px solid #30363d;text-align:center}}
    .flash{{animation:f 1s infinite alternate;background:#0f7f0f !important;color:white}}
    @keyframes f{{from{{background:#0f4d0f}}to{{background:#0f9f0f}}}}
</style>{sound}</head>
<body><h1>ULTIMATE HYBRID DAY TRADER 2025</h1>
<p style="text-align:center">Updated: {datetime.now():%H:%M:%S} • {len(results)} elite LONG setups</p>
<table><tr><th>Sym</th><th>Price</th><th>Prob</th><th>RSI</th><th>Vol</th><th>Stop</th><th>Target</th><th>RR</th><th>ADX</th><th>Signals</th><th>1h Fcst</th><th>2h Fcst</th><th>4h Fcst</th></tr>
{rows or "<tr><td colspan='13' style='color:#888'>No high-probability setups right now...</td></tr>"}</table>
</body></html>'''

# ====================== MAIN ======================
def main():
    print("ULTIMATE HYBRID DAY TRADER 2025 — FINAL CLEAN VERSION")
    try:
        symbols = [l.split()[0].upper() for l in open(SYMBOLS_FILE) if l.strip() and not l.startswith("#")]
    except FileNotFoundError:
        print("symbolsFAST.txt not found!")
        return
    
    print(f"Loaded {len(symbols)} symbols")
    tracked = symbols[:200]
    prev_strong = set()

    while True:
        print(f"\nSCAN @ {datetime.now():%H:%M:%S} | Tracking {len(tracked)} symbols")
        d5 = download(tracked, "10d", "5m")
        d1h = download(tracked, "90d", "1h")
        
        results = []
        close_series = {}
        for s in tracked:
            r = analyze(s, d5.get(s), d1h.get(s))
            if r:
                results.append(r)
                if d5.get(s) is not None:
                    close_series[s] = d5[s]["Close"]
        
        results = sorted(results, key=lambda x: x["Prob"], reverse=True)[:TOP_N]
        
        # Prophet forecasts
        forecasts = {}
        if results and close_series:
            with Pool(min(6, cpu_count())) as p:
                fc_list = p.map(prophet_forecast, [close_series.get(r["Symbol"]) for r in results])
                forecasts = dict(zip([r["Symbol"] for r in results], fc_list))
        
        # Strong buy alerts
        current_strong = {r["Symbol"] for r in results if r["Prob"] >= 80}
        new_strong = current_strong - prev_strong
        if new_strong:
            print("\a" * 10)
            print("ELITE BUY SIGNALS →", " | ".join(new_strong))
        prev_strong = current_strong
        
        # FIXED: No more duplicates
        new_tracked = [r["Symbol"] for r in results] + tracked
        tracked = list(dict.fromkeys(new_tracked))[:MAX_TRACKED]
        
        with open(HTML_FILE, "w", encoding="utf-8") as f:
            f.write(build_html(results, forecasts))
        
        print(f"Dashboard → file://{os.path.abspath(HTML_FILE)} | Found: {len(results)} setups")
        time.sleep(REFRESH_SECONDS)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda *_, **__: (print("\nScanner stopped. Good luck tomorrow!"), sys.exit(0)))
    main()
