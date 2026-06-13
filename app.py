import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
from scipy.stats import norm
from scipy.interpolate import griddata

OPTIONABLE_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "AMD", "NFLX", "JPM", "BAC", "GS", "XOM", "CVX", "SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "SLV", "USO"]

INSUFFICIENT_LIQUIDITY_MESSAGE = ("Not enough liquid option quotes are available for this selection. " "Please choose a more actively traded strike, maturity, or option type.")

def get_last_price(ticker):

    try:
        spot = ticker.fast_info["lastPrice"]
    except Exception:
        try:
            history = ticker.history(period="5d")
            spot = history["Close"].dropna().iloc[-1] if not history.empty else np.nan
        except Exception:
            spot = np.nan

    try:
        spot = float(spot)
    except (TypeError, ValueError):
        return np.nan

    return spot if np.isfinite(spot) and spot > 0 else np.nan

def find_data(ticker,option):

    dfs = []
    maturities = ticker.options
    today = datetime.today()

    if len(maturities) == 0:
        return pd.DataFrame(columns=["strike", "bid", "ask", "currency", "option_type", "maturity", "mid"])

    for maturity in maturities:

        maturity_datetime = datetime.strptime(maturity, "%Y-%m-%d")

        if maturity_datetime <= today + timedelta(days=1):
            
            continue

        try:
            chain = ticker.option_chain(maturity)
        except Exception:
            continue

        if option == 'Calls':

            df = pd.DataFrame(chain.calls)
            df["option_type"] = "Calls"
        
        elif option == 'Puts':

            df = pd.DataFrame(chain.puts)
            df["option_type"] = "Puts"

        elif option == 'Calls & Puts':

            df_calls = pd.DataFrame(chain.calls)
            df_calls["option_type"] = "Calls"
            df_puts = pd.DataFrame(chain.puts)
            df_puts["option_type"] = "Puts"
            df = pd.concat([df_calls, df_puts], ignore_index=True)

        if df.empty:
            continue

        df = df[["strike","bid","ask","currency","option_type"]]
        df["maturity"] = maturity
        dfs.append(df)
    
    if len(dfs) == 0:
        return pd.DataFrame(columns=["strike", "bid", "ask", "currency", "option_type", "maturity", "mid"])

    data = pd.concat(dfs, ignore_index=True)
    data["mid"] = (data["bid"] + data["ask"])/2
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["strike", "bid", "ask", "mid"])
    data = data[data["mid"] > 0]
    
    return data

def risk_free_rate(data):

    if data.empty:
        return np.nan
    
    currency = data["currency"].iloc[0]
    rf = np.nan
    
    if currency == "USD":

        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SOFR"
        df = pd.read_csv(url)
        df = df[df["SOFR"] != "."]
        rf = float(df["SOFR"].iloc[-1]) / 100
    
    if currency == "EUR":
        
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=ECBESTRVOLWGTTRMDMNRT"
        df = pd.read_csv(url)
        df = df[df["ECBESTRVOLWGTTRMDMNRT"] != "."]
        rf = float(df["ECBESTRVOLWGTTRMDMNRT"].iloc[-1]) / 100
    
    if currency == "GBP":
        url = "https://www.bankofengland.co.uk/boeapps/database/fromshowcolumns.asp?CSVF=TT&DAT=RNG&FD=1&FM=Jan&FY=2024&TD=31&TM=Dec&TY=2030&FNY=&Filter=N&FromSeries=1&ToSeries=50&SeriesCodes=IUDSOIA&UsingCodes=Y&VPD=Y"
        df = pd.read_csv(url)
        df = df[df["IUDSOIA"] != "."]
        rf = float(df["IUDSOIA"].iloc[-1]) / 100
    
    return rf

def pricer_bs(strike,maturity,IV,spot,rf,option_type):

    today = datetime.today()
    T = (maturity - today).days / 365.0

    if T <= 0 or IV <= 0 or spot <= 0 or strike <= 0 or not np.isfinite(rf):
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    d1 = (np.log(spot/strike) + (rf + 0.5 * IV**2) * T) / (IV * np.sqrt(T))
    d2 = d1 - IV * np.sqrt(T)

    if option_type == 'Calls':
        
        price = spot * norm.cdf(d1) - strike * np.exp(-rf * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (-(spot * norm.pdf(d1) * IV) / (2 * np.sqrt(T)) - rf * strike * np.exp(-rf * T) * norm.cdf(d2)) / 365.0
        rho = strike * T * np.exp(-rf * T) * norm.cdf(d2)
    
    if option_type == 'Puts':

        price = strike * np.exp(-rf * T) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        theta = (-(spot * norm.pdf(d1) * IV) / (2 * np.sqrt(T)) + rf * strike * np.exp(-rf * T) * norm.cdf(-d2)) / 365.0
        rho = -strike * T * np.exp(-rf * T) * norm.cdf(-d2)

    gamma = norm.pdf(d1) / (spot * IV * np.sqrt(T))
    vega = spot * norm.pdf(d1) * np.sqrt(T)
    volga = vega * d1 * d2 / IV
    vanna = -norm.pdf(d1) * d2 / IV

    return price, delta, gamma, vega, theta, rho, volga, vanna

def newton_raphson(ticker,data,initial_IV,spot,rf,max_iter=100):
    
    IV = initial_IV
    mid = data["mid"]
    option_type = data["option_type"]
    strike = data["strike"]
    maturity = datetime.strptime(data["maturity"], "%Y-%m-%d")

    price, delta, gamma, vega, theta, rho, volga, vanna = pricer_bs(strike, maturity, IV, spot, rf, option_type)

    for _ in range(max_iter):

        if not np.isfinite(price) or not np.isfinite(vega) or not np.isfinite(IV):
            return np.nan

        if abs(price - mid) <= 1e-5:
            return IV

        if abs(vega) < 1e-8:
            return np.nan

        IV = IV - (price - mid) / vega

        if IV <= 0:
            return np.nan

        price, delta, gamma, vega, theta, rho, volga, vanna = pricer_bs(strike, maturity, IV, spot, rf, option_type)

    return np.nan

def term_structure(ticker,option,strike,maturities=None,return_fig=False):
    
    ticker = yf.Ticker(ticker)
    data_strike = find_data(ticker, option)
    data_strike = data_strike[data_strike["strike"] == strike].copy()

    if maturities is not None and len(maturities) > 0:
        data_strike = data_strike[data_strike["maturity"].isin(maturities)].copy()

    if data_strike.empty:
        
        return np.nan

    spot = get_last_price(ticker)

    if not np.isfinite(spot):
        return np.nan

    rf = risk_free_rate(data_strike)

    data_strike["model_IV"] = data_strike.apply(lambda row: newton_raphson(ticker,row,0.2,spot,rf), axis=1)
    data_strike = data_strike[["strike","maturity","model_IV"]].dropna()

    if data_strike.empty:
        return np.nan

    data_strike["maturity_date"] = pd.to_datetime(data_strike["maturity"])

    plt.style.use("dark_background")

    fig, ax = plt.subplots(figsize=(12, 6), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")

    x = data_strike["maturity_date"]
    y = data_strike["model_IV"] * 100

    ax.plot(x, y,color="#00d4ff",linewidth=2.5,marker="o",markersize=7,markerfacecolor="#0e1117",markeredgecolor="#00d4ff",markeredgewidth=2)

    ax.fill_between(x, y,color="#00d4ff",alpha=0.12)

    ax.set_title(f"Implied Volatility Term Structure\n{option.upper()} | Strike {strike}",fontsize=16,fontweight="bold",color="white",pad=18)

    ax.set_xlabel("Maturity", fontsize=12, color="#d0d0d0")
    ax.set_ylabel("Implied volatility (%)", fontsize=12, color="#d0d0d0")

    ax.grid(True,linestyle="--",linewidth=0.6,alpha=0.25,color="white")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
    plt.xticks(rotation=45, ha="right")

    ax.tick_params(axis="x", colors="#d0d0d0")
    ax.tick_params(axis="y", colors="#d0d0d0")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")

    for xi, yi in zip(x, y):ax.annotate(f"{yi:.1f}%",xy=(xi, yi),xytext=(0, 8),textcoords="offset points",ha="center",fontsize=9,color="#e6e6e6")

    plt.tight_layout()

    if not return_fig:
        plt.show()

    data_strike = data_strike[["strike","maturity","model_IV"]]
    return (data_strike, fig) if return_fig else data_strike

def skew(ticker,option,maturity,strikes=None,return_fig=False):
    
    ticker = yf.Ticker(ticker)
    data_maturity = find_data(ticker, option)
    data_maturity = data_maturity[data_maturity["maturity"] == maturity].copy()

    if strikes is not None and len(strikes) > 0:
        data_maturity = data_maturity[data_maturity["strike"].isin(strikes)].copy()

    if data_maturity.empty:
        
        return np.nan

    spot = get_last_price(ticker)

    if not np.isfinite(spot):
        return np.nan

    rf = risk_free_rate(data_maturity)

    data_maturity["model_IV"] = data_maturity.apply(lambda row: newton_raphson(ticker,row,0.2,spot,rf), axis=1)
    data_maturity = data_maturity[["strike","maturity","model_IV"]].dropna()

    if data_maturity.empty:
        return np.nan
    
    plt.style.use("dark_background")

    fig, ax = plt.subplots(figsize=(12, 6), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")

    x = data_maturity["strike"]
    y = data_maturity["model_IV"] * 100

    ax.plot(x, y,color="#00d4ff",linewidth=2.5,marker="o",markersize=7,markerfacecolor="#0e1117",markeredgecolor="#00d4ff",markeredgewidth=2)

    ax.fill_between(x, y,color="#00d4ff",alpha=0.12)

    ax.set_title(f"Implied Volatility Skew\n{option.upper()} | Maturity {maturity}",fontsize=16,fontweight="bold",color="white",pad=18)

    ax.set_xlabel("Strike", fontsize=12, color="#d0d0d0")
    ax.set_ylabel("Implied volatility (%)", fontsize=12, color="#d0d0d0")

    ax.grid(True,linestyle="--",linewidth=0.6,alpha=0.25,color="white")

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f"{val:.0f}"))
    plt.xticks(rotation=45, ha="right")

    ax.tick_params(axis="x", colors="#d0d0d0")
    ax.tick_params(axis="y", colors="#d0d0d0")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")

    for xi, yi in zip(x, y):ax.annotate(f"{yi:.1f}%",xy=(xi, yi),xytext=(0, 8),textcoords="offset points",ha="center",fontsize=9,color="#e6e6e6")

    plt.tight_layout()

    if not return_fig:
        plt.show()

    return (data_maturity, fig) if return_fig else data_maturity

def volatility_surface(ticker, option, maturities=None, strikes=None, return_fig=False):

    ticker = yf.Ticker(ticker)
    data_surface = find_data(ticker, option)

    if data_surface.empty:
        return np.nan

    if maturities is not None and len(maturities) > 0:
        data_surface = data_surface[data_surface["maturity"].isin(maturities)].copy()

    if strikes is not None and len(strikes) > 0:
        data_surface = data_surface[data_surface["strike"].isin(strikes)].copy()

    if data_surface.empty:
        return np.nan

    spot = get_last_price(ticker)

    if not np.isfinite(spot):
        return np.nan

    if option == "Calls & Puts":
        data_surface = data_surface[
            ((data_surface["option_type"] == "Puts") & (data_surface["strike"] < spot))
            | ((data_surface["option_type"] == "Calls") & (data_surface["strike"] > spot))
        ].copy()

        if data_surface.empty:
            return np.nan

    rf = risk_free_rate(data_surface)

    data_surface["model_IV"] = data_surface.apply(lambda row: newton_raphson(ticker, row, 0.2, spot, rf),axis=1)

    data_surface = data_surface[["strike", "maturity", "option_type", "model_IV"]].copy()

    data_surface["strike"] = pd.to_numeric(data_surface["strike"], errors="coerce")
    data_surface["model_IV"] = pd.to_numeric(data_surface["model_IV"], errors="coerce")
    data_surface["maturity_date"] = pd.to_datetime(data_surface["maturity"], errors="coerce")

    data_surface = data_surface.replace([np.inf, -np.inf], np.nan)
    data_surface = data_surface.dropna(subset=["strike", "maturity_date", "model_IV"])

    if data_surface.empty:
        return np.nan

    today = pd.Timestamp.today().normalize()
    data_surface["T_years"] = ((data_surface["maturity_date"] - today).dt.days / 365)

    data_surface = data_surface[data_surface["T_years"] > 0]

    if data_surface.empty:
        return np.nan

    data_surface["moneyness"] = data_surface["strike"] / spot
    data_surface["IV_percent"] = data_surface["model_IV"] * 100

    lower_iv = np.nanpercentile(data_surface["IV_percent"], 2)
    upper_iv = np.nanpercentile(data_surface["IV_percent"], 98)

    data_surface = data_surface[(data_surface["IV_percent"] > lower_iv) &(data_surface["IV_percent"] < upper_iv)]

    if data_surface.empty:
        return np.nan

    x = data_surface["moneyness"].values
    y = data_surface["T_years"].values
    z = data_surface["IV_percent"].values

    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    X, Y = np.meshgrid(xi, yi)

    try:
        Z = griddata(points=(x, y),values=z,xi=(X, Y),method="linear")
    except Exception:
        Z = np.full_like(X, np.nan, dtype=float)

    Z_nearest = griddata(points=(x, y),values=z,xi=(X, Y),method="nearest")

    Z = np.where(np.isnan(Z), Z_nearest, Z)

    z_min = np.nanmin(Z)
    z_max = np.nanpercentile(Z, 99)

    def build_surface_figure(azim):

        plt.style.use("dark_background")

        fig = plt.figure(figsize=(15, 9), facecolor="#0e1117")
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("#0e1117")

        cyan_cmap = LinearSegmentedColormap.from_list("cyan_vol_surface",["#001f26", "#00d4ff", "#e6fbff"])

        surface = ax.plot_surface(X,Y,Z,cmap=cyan_cmap,linewidth=0,antialiased=True,alpha=0.95,rstride=1,cstride=1)

        title_option = "OTM PUTS / CALLS" if option == "Calls & Puts" else option.upper()
        ax.set_title(f"Implied Volatility Surface\n{title_option}",fontsize=18,fontweight="bold",color="white",pad=24)

        ax.set_xlabel("Moneyness K/S", fontsize=12, color="#d0d0d0", labelpad=12)
        ax.set_ylabel("Maturity in years", fontsize=12, color="#d0d0d0", labelpad=12)
        ax.set_zlabel("Implied volatility (%)", fontsize=12, color="#d0d0d0", labelpad=12)

        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{val:.2f}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{val:.1f}Y"))
        ax.zaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{val:.1f}%"))

        ax.tick_params(axis="x", colors="#d0d0d0", labelsize=9)
        ax.tick_params(axis="y", colors="#d0d0d0", labelsize=9)
        ax.tick_params(axis="z", colors="#d0d0d0", labelsize=9)

        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.pane.set_facecolor("#0e1117")
            axis.pane.set_edgecolor("#555555")
            axis._axinfo["grid"]["color"] = (1, 1, 1, 0.15)
            axis._axinfo["grid"]["linestyle"] = "--"
            axis._axinfo["grid"]["linewidth"] = 0.5

        ax.set_proj_type("ortho")
        ax.set_box_aspect((1.6, 1.1, 0.7))

        ax.view_init(elev=22, azim=azim)
        ax.set_zlim(z_min, z_max)

        cbar = fig.colorbar(surface, ax=ax, shrink=0.65, pad=0.08)
        cbar.set_label("Implied volatility (%)", color="#d0d0d0", fontsize=11)
        cbar.ax.yaxis.set_tick_params(color="#d0d0d0")
        plt.setp(cbar.ax.get_yticklabels(), color="#d0d0d0")

        plt.tight_layout()

        return fig

    fig = build_surface_figure(azim=35)
    opposite_fig = build_surface_figure(azim=215)

    if not return_fig:
        plt.show()
        plt.figure(opposite_fig.number)
        plt.show()

    return (data_surface, [fig, opposite_fig]) if return_fig else data_surface

def get_available_maturities(symbol):

    ticker = yf.Ticker(symbol)
    today = datetime.today()
    maturities = []

    for maturity in ticker.options:
        maturity_datetime = datetime.strptime(maturity, "%Y-%m-%d")

        if maturity_datetime > today + timedelta(days=1):
            maturities.append(maturity)

    return maturities

def load_option_data(symbol, option_type):

    ticker = yf.Ticker(symbol)
    return find_data(ticker, option_type)

def load_spot_price(symbol):

    ticker = yf.Ticker(symbol)
    return get_last_price(ticker)

def format_strike(strike):

    return f"{strike:.2f}".rstrip("0").rstrip(".")

def format_spot_price(spot, currency):

    if not np.isfinite(spot):
        return "N/A"

    currency_suffix = f" {currency}" if isinstance(currency, str) and len(currency) > 0 else ""
    return f"{spot:,.2f}{currency_suffix}"

def streamlit_app():

    st.set_page_config(page_title="Implied Volatility Dashboard", layout="wide")

    st.title("Implied Volatility Dashboard")
    st.caption("Select a ticker, then choose strikes and maturities available in the Yahoo Finance option chain.")

    with st.sidebar:
        st.header("Settings")

        ticker_source = st.radio("Ticker",["Liquid ticker universe", "Custom ticker"],horizontal=False)

        if ticker_source == "Liquid ticker universe":
            default_index = OPTIONABLE_TICKERS.index("AAPL") if "AAPL" in OPTIONABLE_TICKERS else 0
            symbol = st.selectbox("Ticker", OPTIONABLE_TICKERS, index=default_index)
        else:
            symbol = st.text_input("Enter a Yahoo Finance ticker", value="AAPL")

        symbol = symbol.strip().upper()

        option_type = st.selectbox("Option type", ["Calls", "Puts", "Calls & Puts"], index=0)

        view_type = st.radio("Analysis",["Skew", "Term structure", "Volatility surface"],horizontal=False)

    if not symbol:
        st.info("Enter a ticker to begin.")
        return

    try:
        maturities = get_available_maturities(symbol)
    except Exception as error:
        st.error(f"Unable to retrieve option maturities for {symbol}: {error}")
        return

    if len(maturities) == 0:
        st.warning(f"No option maturities are available for {symbol} via Yahoo Finance.")
        return

    with st.spinner("Loading option chain..."):
        data = load_option_data(symbol, option_type)

    if data.empty:
        st.warning("No usable option quotes remain after filtering bid, ask, and mid prices.")
        return

    spot_price = load_spot_price(symbol)
    display_currency = data["currency"].dropna().iloc[0] if not data["currency"].dropna().empty else ""
    available_maturities = sorted(data["maturity"].dropna().unique())
    available_strikes = sorted(data["strike"].dropna().unique())

    col_metric_1, col_metric_2, col_metric_3, col_metric_4 = st.columns(4)
    col_metric_1.metric("Spot price", format_spot_price(spot_price, display_currency))
    col_metric_2.metric("Available maturities", len(available_maturities))
    col_metric_3.metric("Available strikes", len(available_strikes))
    col_metric_4.metric("Loaded options", len(data))

    st.subheader(f"{symbol} - {view_type}")

    if view_type == "Skew":
        maturity = st.selectbox("Available maturity", available_maturities)
        strikes_for_maturity = sorted(data.loc[data["maturity"] == maturity, "strike"].dropna().unique())
        selected_strikes = st.multiselect("Available strikes for this maturity",strikes_for_maturity,default=strikes_for_maturity,format_func=format_strike)

        if option_type == "Calls & Puts":
            st.info("For a cleaner skew view, select either calls or puts in the sidebar.")

        if st.button("Display skew", type="primary"):
            with st.spinner("Calculating implied volatility..."):
                result = skew(symbol, option_type, maturity, strikes=selected_strikes, return_fig=True)

            if isinstance(result, tuple):
                result_data, fig = result
                st.pyplot(fig, clear_figure=True)
                st.dataframe(result_data, use_container_width=True)
            else:
                st.warning(INSUFFICIENT_LIQUIDITY_MESSAGE)

    elif view_type == "Term structure":
        strike = st.selectbox("Available strike",available_strikes,format_func=format_strike)
        maturities_for_strike = sorted(data.loc[data["strike"] == strike, "maturity"].dropna().unique())
        selected_maturities = st.multiselect("Available maturities for this strike",maturities_for_strike,default=maturities_for_strike)

        if option_type == "Calls & Puts":
            st.info("For a cleaner term structure view, select either calls or puts in the sidebar.")

        if st.button("Display term structure", type="primary"):
            with st.spinner("Calculating implied volatility..."):
                result = term_structure(symbol, option_type, strike, maturities=selected_maturities, return_fig=True)

            if isinstance(result, tuple):
                result_data, fig = result
                st.pyplot(fig, clear_figure=True)
                st.dataframe(result_data, use_container_width=True)
            else:
                st.warning(INSUFFICIENT_LIQUIDITY_MESSAGE)

    else:
        selected_maturities = st.multiselect("Available maturities",available_maturities,default=available_maturities)
        selected_strikes = st.multiselect("Available strikes",available_strikes,default=available_strikes,format_func=format_strike)

        if st.button("Display volatility surface", type="primary"):
            with st.spinner("Calculating implied volatility surface..."):
                result = volatility_surface(symbol,option_type,maturities=selected_maturities,strikes=selected_strikes,return_fig=True)

            if isinstance(result, tuple):
                result_data, figures = result
                for fig in figures:
                    st.pyplot(fig, clear_figure=True)
                st.dataframe(result_data, use_container_width=True)
            else:
                st.warning(INSUFFICIENT_LIQUIDITY_MESSAGE)

if __name__ == "__main__":
    streamlit_app()
