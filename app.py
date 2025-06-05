import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import argrelextrema
import datetime

st.set_page_config(layout="wide")

st.title("NIFTY Intraday Trading with PDH, PDL, PDC, EMA20 & Classic Patterns")

@st.cache_data(ttl=300)
def load_data():
    ticker = "^NSEI"
    today = datetime.date.today()
    start = today - datetime.timedelta(days=5)
    df = yf.download(ticker, interval='1m', start=start, progress=False)
    df.reset_index(inplace=True)
    df.rename(columns={'Datetime':'Datetime'}, inplace=True)
    return df

df = load_data()

df['date'] = df['Datetime'].dt.date
today = datetime.date.today()
intraday = df[df['date'] == today].copy()
intraday.reset_index(drop=True, inplace=True)

if intraday.empty:
    st.warning("No intraday data available for today yet. Please try after market hours or refresh later.")
    st.stop()

prev_day = today - datetime.timedelta(days=1)
prev_day_data = df[df['date'] == prev_day]

if prev_day_data.empty:
    st.warning("No previous day data available for reference levels.")
    st.stop()

PDH = prev_day_data['High'].max()
PDL = prev_day_data['Low'].min()
PDC = prev_day_data['Close'].iloc[-1]

st.markdown(f"**Previous Day High (PDH):** {PDH:.2f} | **Previous Day Low (PDL):** {PDL:.2f} | **Previous Day Close (PDC):** {PDC:.2f}")

intraday['EMA20'] = intraday['Close'].ewm(span=20, adjust=False).mean()

def is_inside_candle(df):
    inside = []
    for i in range(1, len(df)):
        prev_high = df.loc[i-1, 'High']
        prev_low = df.loc[i-1, 'Low']
        curr_high = df.loc[i, 'High']
        curr_low = df.loc[i, 'Low']
        if curr_high < prev_high and curr_low > prev_low:
            inside.append(True)
        else:
            inside.append(False)
    inside.insert(0, False)
    return inside

intraday['Inside_Candle'] = is_inside_candle(intraday)

def detect_candlestick_patterns(df):
    patterns = ['']*len(df)
    for i in range(1, len(df)):
        o, h, l, c = df.loc[i, ['Open','High','Low','Close']]
        o1, c1 = df.loc[i-1, ['Open','Close']]
        body = abs(c - o)
        body_prev = abs(c1 - o1)
        upper_shadow = h - max(o,c)
        lower_shadow = min(o,c) - l

        if body <= (h - l)*0.1:
            patterns[i] = 'Doji'
        elif c > o and c1 < o1 and c > o1 and o < c1:
            patterns[i] = 'Bullish Engulfing'
        elif c < o and c1 > o1 and c < o1 and o > c1:
            patterns[i] = 'Bearish Engulfing'
        elif body / (h - l) < 0.4 and lower_shadow > 2 * body and upper_shadow < body:
            if c > o:
                patterns[i] = 'Hammer'
        elif body / (h - l) < 0.4 and upper_shadow > 2 * body and lower_shadow < body:
            if c < o:
                patterns[i] = 'Shooting Star'
    return patterns

intraday['Candlestick_Pattern'] = detect_candlestick_patterns(intraday)

from scipy.signal import argrelextrema
def find_local_extrema(prices, order=5):
    local_max = argrelextrema(prices.values, np.greater_equal, order=order)[0]
    local_min = argrelextrema(prices.values, np.less_equal, order=order)[0]
    return local_max, local_min

def approx_equal(a, b, tol=0.005):
    return abs(a - b) <= tol * a

def detect_double_top(prices, local_max):
    patterns = []
    for i in range(len(local_max) -1):
        p1, p2 = local_max[i], local_max[i+1]
        if approx_equal(prices[p1], prices[p2]):
            valley = prices[p1:p2].min()
            if valley < prices[p1] * 0.98:
                patterns.append((p1, p2, "Double Top"))
    return patterns

def detect_double_bottom(prices, local_min):
    patterns = []
    for i in range(len(local_min) -1):
        t1, t2 = local_min[i], local_min[i+1]
        if approx_equal(prices[t1], prices[t2]):
            peak = prices[t1:t2].max()
            if peak > prices[t1] * 1.02:
                patterns.append((t1, t2, "Double Bottom"))
    return patterns

def detect_head_shoulders(prices, local_max):
    patterns = []
    for i in range(len(local_max) - 2):
        left, head, right = local_max[i], local_max[i+1], local_max[i+2]
        if prices[head] > prices[left] and prices[head] > prices[right]:
            if approx_equal(prices[left], prices[right]):
                patterns.append((left, head, right, "Head and Shoulders"))
    return patterns

def detect_inverse_head_shoulders(prices, local_min):
    patterns = []
    for i in range(len(local_min) - 2):
        left, head, right = local_min[i], local_min[i+1], local_min[i+2]
        if prices[head] < prices[left] and prices[head] < prices[right]:
            if approx_equal(prices[left], prices[right]):
                patterns.append((left, head, right, "Inverse Head and Shoulders"))
    return patterns

def detect_classic_patterns(df):
    close = df['Close']
    local_max, local_min = find_local_extrema(close, order=5)
    
    patterns = []

    dt = df['Datetime']

    for p1, p2, name in detect_double_top(close, local_max):
        patterns.append({
            "Pattern": name,
            "Points": [dt.iloc[p1], dt.iloc[p2]],
            "Prices": [close.iloc[p1], close.iloc[p2]],
            "Index": [p1, p2]
        })

    for t1, t2, name in detect_double_bottom(close, local_min):
        patterns.append({
            "Pattern": name,
            "Points": [dt.iloc[t1], dt.iloc[t2]],
            "Prices": [close.iloc[t1], close.iloc[t2]],
            "Index": [t1, t2]
        })

    for left, head, right, name in detect_head_shoulders(close, local_max):
        patterns.append({
            "Pattern": name,
            "Points": [dt.iloc[left], dt.iloc[head], dt.iloc[right]],
            "Prices": [close.iloc[left], close.iloc[head], close.iloc[right]],
            "Index": [left, head, right]
        })

    for left, head, right, name in detect_inverse_head_shoulders(close, local_min):
        patterns.append({
            "Pattern": name,
            "Points": [dt.iloc[left], dt.iloc[head], dt.iloc[right]],
            "Prices": [close.iloc[left], close.iloc[head], close.iloc[right]],
            "Index": [left, head, right]
        })

    return patterns

classic_patterns = detect_classic_patterns(intraday)

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=intraday['Datetime'],
    open=intraday['Open'], high=intraday['High'],
    low=intraday['Low'], close=intraday['Close'],
    name='Price'
))

fig.add_trace(go.Scatter(
    x=intraday['Datetime'], y=intraday['EMA20'],
    mode='lines', name='EMA 20', line=dict(color='blue')
))

fig.add_hline(y=PDH, line_dash="dash", line_color="green", annotation_text="PDH", annotation_position="top left")
fig.add_hline(y=PDL, line_dash="dash", line_color="red", annotation_text="PDL", annotation_position="bottom left")
fig.add_hline(y=PDC, line_dash="dot", line_color="orange", annotation_text="PDC", annotation_position="bottom right")

for i, row in intraday.iterrows():
    if row['Candlestick_Pattern']:
        fig.add_trace(go.Scatter(
            x=[row['Datetime']], y=[row['High']],
            mode='markers+text',
            marker=dict(symbol='star', size=10, color='purple'),
            text=[row['Candlestick_Pattern']],
            textposition="top center",
            showlegend=False,
            hoverinfo='text'
        ))

inside_candle_points = intraday[intraday['Inside_Candle']]
fig.add_trace(go.Scatter(
    x=inside_candle_points['Datetime'], y=inside_candle_points['High'] + 5,
    mode='markers',
    marker=dict(symbol='circle', size=8, color='cyan'),
    name='Inside Candle'
))

for p in classic_patterns:
    if p['Pattern'] in ['Double Top', 'Double Bottom']:
        fig.add_trace(go.Scatter(
            x=p['Points'], y=p['Prices'],
            mode='lines+markers+text',
            line=dict(color='magenta', width=3, dash='dash'),
            marker=dict(size=10, color='magenta'),
            text=[p['Pattern']] * len(p['Points']),
            textposition="bottom center",
            name=p['Pattern']
        ))
    else:
        fig.add_trace(go.Scatter(
            x=p['Points'], y=p['Prices'],
            mode='lines+markers+text',
            line=dict(color='orange', width=3, dash='dot'),
            marker=dict(size=10, color='orange'),
            text=[p['Pattern']] * len(p['Points']),
            textposition="bottom center",
            name=p['Pattern']
        ))

fig.update_layout(xaxis_rangeslider_visible=False, height=700, title="NIFTY Intraday with Patterns")

st.plotly_chart(fig, use_container_width=True)

if classic_patterns:
    st.subheader("Detected Classic Patterns Today")
    rows = []
    for p in classic_patterns:
        points_str = ", ".join([dt.strftime("%H:%M") for dt in p['Points']])
        prices_str = ", ".join([f"{price:.2f}" for price in p['Prices']])
        rows.append({
            "Pattern": p['Pattern'],
            "Points (Time)": points_str,
            "Prices": prices_str
        })
    st.table(pd.DataFrame(rows))
else:
    st.info("No classic patterns detected for today.")

intraday['Position'] = 0
intraday.loc[(intraday['Close'] > intraday['EMA20']) & (intraday['Close'] > PDH), 'Position'] = 1
intraday.loc[(intraday['Close'] < intraday['EMA20']) | (intraday['Close'] < PDH), 'Position'] = 0

intraday['Entry_Signal'] = (intraday['Position'].diff() == 1)
intraday['Exit_Signal'] = (intraday['Position'].diff() == -1)

fig.add_trace(go.Scatter(
    x=intraday.loc[intraday['Entry_Signal'], 'Datetime'],
    y=intraday.loc[intraday['Entry_Signal'], 'Low'] * 0.995,
    mode='markers',
    marker=dict(symbol='triangle-up', size=12, color='green'),
    name='Entry Signal'
))

fig.add_trace(go.Scatter(
    x=intraday.loc[intraday['Exit_Signal'], 'Datetime'],
    y=intraday.loc[intraday['Exit_Signal'], 'High'] * 1.005,
    mode='markers',
    marker=dict(symbol='triangle-down', size=12, color='red'),
    name='Exit Signal'
))

st.plotly_chart(fig, use_container_width=True)
