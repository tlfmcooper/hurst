# Importing Libraries
from urllib.parse import non_hierarchical
import investpy as inv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hurst import compute_Hc
from typing import Optional
import requests
from math import floor
from termcolor import colored as cl
from scipy import signal
import contextlib
import bt
import talib
import talib as ta
import sklearn
from sklearn.model_selection import (
    TimeSeriesSplit,
    train_test_split, GridSearchCV
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
# Either update scikit learn or use this line also add this in ' ./hurst_utils/utils/utils.py'
from sklearn.experimental import enable_hist_gradient_boosting # this line
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import matplotlib.pyplot as plt


def get_index_close_price(
    from_date: str,
    to_date: str,
    type: str = "emerging",
    order: Optional[str] = None,
    interval: Optional[str] = None,
    country: Optional[str] = None,
    index: Optional[str] = None,
    crypto: Optional[str] = None,
    text: Optional[str] = None,
) -> pd.DataFrame:

    if type == "crypto":
        return inv.crypto.get_crypto_historical_data(
            crypto=crypto,
            from_date=from_date,
            to_date=to_date,
            order=order,
            interval=interval,
        )["Close"]

    elif type == "index":
        return inv.indices.get_index_historical_data(
            index=index,
            country=country,
            from_date=from_date,
            to_date=to_date,
            order=order,
            interval=interval,
        )["Close"]

    else:
        index_search = inv.search.search_quotes(text=text, n_results=1)
        return index_search.retrieve_historical_data(
            from_date=from_date, to_date=to_date
        )["Close"]


def get_index_price(
    from_date: str,
    to_date: str,
    type: str = "emerging",
    order: Optional[str] = None,
    interval: Optional[str] = None,
    country: Optional[str] = None,
    index: Optional[str] = None,
    crypto: Optional[str] = None,
    text: Optional[str] = None,
    close_only: bool = True,
) -> pd.DataFrame:

    if close_only:
        return get_index_close_price(
            from_date, to_date, type, order, interval, country, index, crypto, text
        )
    if type == "crypto":
        return inv.crypto.get_crypto_historical_data(
            crypto=crypto,
            from_date=from_date,
            to_date=to_date,
            order=order,
            interval=interval,
        ).iloc[:, :5]

    elif type == "index":
        return inv.indices.get_index_historical_data(
            index=index,
            country=country,
            from_date=from_date,
            to_date=to_date,
            order=order,
            interval=interval,
        ).iloc[:, :5]

    else:
        index_search = inv.search.search_quotes(text=text, n_results=1)
        return index_search.retrieve_historical_data(
            from_date=from_date, to_date=to_date
        ).iloc[:, :5]
    
# name is a string we want to call our H value column of dataframe
def Hursts(series,name):
    
    # We require this step since Hurst function requires atleast 100 data points
    div = int(len(series)/100)
    # We use "dtype = int" to ensure the data type of output generated is integer for further slicing later
    price_series = np.linspace(0,len(series),div, dtype = int)
    
    H_list = []
    for i in range(1,len(price_series)):
       # temp = compute_Hc(series[price_series[i-1]:price_series[i]],kind = 'price')[0]
        temp = compute_Hc(series[price_series[i-1]:price_series[i]],min_window = 16,max_window = 32)[0]
        H_list.append(temp)
    
    '''h_series = pd.Series(H_list) ''' # This step isn't required
    
    # Create an empty series with the same index as our Financial Time Series
    df = pd.DataFrame(data = pd.Series( dtype = 'float64'), index = series.index)
    
    # Making H value dataframe and applyin forward filling since Hurst value is for the entire interval
    for j in range(1,len(price_series)):
        # using the index, we will get the value of price series at that index
        df.iloc[price_series[j-1],0] = H_list[j-1] #h_series[j-1]
        
    df.rename(columns = {0:'H_value'},inplace = True)
    df = df.add_suffix('_'+name)
    df = df.ffill(axis = 0)
    
    return df

def signal_int(series,hurst_df, ema_s, ema_l,series_name):
   
    # series = Financial time Series 
    # hurst_df is our Curst computed DataFrame
    # ema_s is our small moving average
    # ema_l is our large moving average
    # series_name is the name of our financial time series 
    
    # Computing Moving averages
    ema_short = talib.EMA(series, timeperiod = ema_s).to_frame()
    ema_long = talib.EMA(series, timeperiod = ema_l).to_frame()
    
    # Signal based on Moving Average
    
    # Create the signal DataFrame
    signal = ema_long.copy()
    signal[ema_long.isnull()] = 0
    # Construct the signal based on moving avg
    signal[ema_short > ema_long] = 1
    signal[ema_short < ema_long] = -1
    
    # Signal based on Hurst Exponent
    signal_1 = hurst_df.copy()
    #np.logical_and(signal_1['H_value']<0.55,signal_1['H_value']>0.45) = 0 #and signal_1['H_value'] >0.45
    signal_1[signal_1>0.50] = 1
    signal_1[signal_1<0.50] = -1
    
    # it is important to rename the signal columns the same as our original dataframe
    signal.rename(columns = {0:series_name}, inplace = True)
    signal_1.rename(columns = {hurst_df.columns[0]: series_name}, inplace = True)
    
    # Combined Signal
    signal_comb = signal*signal_1
    signal_comb[signal_comb == -0] = 0
    
    # Plotting Financial series with Signals and Moving Averages
    combo_df = bt.merge(signal_comb,series,ema_short,ema_long)
    combo_df.columns = ['Signal Strategy',series_name,'EMA_short','EMA_LONG']
    ax_name = combo_df.columns[0]
    
    # Applying Strategy
    # Creating DataFrame of Financial Time Series
    df = pd.DataFrame(series)
    df.rename(columns = {'Close' : series_name}, inplace = True)
    
    # Running Strategy Strategy
    bt_strategy_cross = bt.Strategy('EMA_crossover with Hurst for '+series_name,\
                                    [bt.algos.WeighTarget(signal_comb),bt.algos.Rebalance()])
    bt_backtest_cross = bt.Backtest(bt_strategy_cross, df)
    bt_result_cross = bt.run(bt_backtest_cross)
    
    return bt_result_cross, combo_df, ax_name

# Here we are Defining our Buy and Hold strategy
def buy_and_hold(series, name): 
    # Get the data
    # Creating DataFrame of Financial Time Series
    df = pd.DataFrame(series)
    df.rename(columns = {'Close' : name}, inplace = True)

    # Define the benchmark strategy
    bt_strategy = bt.Strategy(name,[bt.algos.RunOnce(),bt.algos.SelectAll(),bt.algos.WeighEqually(),bt.algos.Rebalance()])
    # Return the backtest
    return bt.Backtest(bt_strategy, df)


def plot_prices(assets: pd.DataFrame):
    kw = dict(figsize=(15, 8), grid=True, subplots=True, layout=(2, 2), linewidth=1)
    axs = assets.plot(**kw)
    [ax.set_ylabel("In dollars ($)") for ax in axs.flat[::2]]
    plt.suptitle("Close Price by Assets", y=0.95)


def plot_rolling_hurst(series: pd.DataFrame):
    kw = dict(figsize=(15, 8), grid=True, subplots=True, layout=(2, 2), linewidth=1)
    axs = series.plot(**kw)
    [
        ax.plot(series.index, np.ones(len(series)) * 0.5, f"C{i}")
        for i, ax in enumerate(axs.flat[:])
    ]
    plt.rc("xtick", labelsize=8)
    plt.suptitle("Hurst Exponent", y=0.95)
    plt.show()


def rolling_hurst(
    series: pd.Series,
    kind="price",
    min_window: int = 10,
    max_window: Optional[int] = None,
    simplified=True,
):
    rollingHursts = []
    for i in range(min_window, len(series)):
        H, _, _ = compute_Hc(
            series,
            kind=kind,
            min_window=i,
            max_window=max_window,
            simplified=simplified,
        )
        rollingHursts.append(H)
    return np.array(rollingHursts)


def ml_feature_engineering(ohlcv: pd.DataFrame) -> pd.DataFrame:

    ohlcv = ohlcv.copy()
    # Calculate the EMA10 > EMA30 signal
    ema10 = ohlcv["Close"].ewm(span=10).mean()
    ema30 = ohlcv["Close"].ewm(span=30).mean()
    ohlcv["EMA10gtEMA30"] = np.where(ema10 > ema30, 1, -1)

    # Calculate where Close is > EMA10
    ohlcv["ClGtEMA10"] = np.where(ohlcv["Close"] > ema10, 1, -1)

    # Calculate MACD
    macd, macd_signal, _ = ta.MACD(ohlcv.Close)
    ohlcv["MACD"] = macd - macd_signal

    # calculate RSI
    ohlcv["RSI"] = ta.RSI(ohlcv.Close)

    # Stochastic Oscillator
    high14 = ohlcv["High"].rolling(14).max()
    low14 = ohlcv["Low"].rolling(14).min()
    ohlcv["%K"] = (ohlcv["Close"] - low14) * 100 / (high14 - low14)

    # Calculate Williams Percentage Range
    ohlcv["WILLR"] = ta.WILLR(ohlcv.High, ohlcv.Low, ohlcv.Close)

    # Calculate Price rate of change
    ohlcv["PROC"] = ta.ROC(ohlcv.Close)
    ohlcv.dropna(inplace=True)

    # Defining the class
    ohlcv["Return"] = ohlcv["Close"].pct_change(1).shift(-1)
    ohlcv["target"] = np.where(ohlcv["Return"] > 0, 1, -1)

    return ohlcv


def take_positions_based_on_model_predictions(pred: np.array):
    # We take positions at the start of the period
    positions = [1]

    for i in range(1, len(pred)):
        # Stay in our position (long)
        if pred[i - 1] == pred[i] == 1:
            positions.append(0)
        # We were not in the market(we sold before), we stay out
        elif pred[i - 1] == pred[i] == -1:
            positions.append(0)

        # we sell or buy depending on the predictions
        else:
            positions.append(pred[i - 1])
    return np.array(positions)


def train_model(ohlcv: pd.DataFrame):
    # Data to predict
    predictors = ["EMA10gtEMA30", "ClGtEMA10", "MACD", "RSI", "%K", "WILLR", "PROC"]
    X = ohlcv[predictors]
    y = ohlcv["target"]
    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, shuffle=False
    )
    # Train the model
    model = RandomForestClassifier(random_state=0)
    model.fit(X_train, y_train)

    return model


def model_predict(model: sklearn.ensemble, ohlcv: pd.DataFrame):
    predictors = ["EMA10gtEMA30", "ClGtEMA10", "MACD", "RSI", "%K", "WILLR", "PROC"]
    X = ohlcv[predictors]
    y = ohlcv["target"]
    # Split data into train and test
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20)
    # See how accurate the predictions are
    return model.predict(X_test)


def train_model_predict(ohlcv: pd.DataFrame):
    # Data to predict
    predictors = ["EMA10gtEMA30", "ClGtEMA10", "MACD", "RSI", "%K", "WILLR", "PROC"]

    X = ohlcv[predictors]
    y = ohlcv["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, shuffle=False
    )
    groups = ohlcv.index.to_period("Q")
    cv = TimeSeriesSplit(n_splits=groups.nunique())

    param_grid = {
        "max_iter": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "max_depth": [3, 5, None],
    }

    grid_search = GridSearchCV(
        HistGradientBoostingClassifier(),
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        cv=cv,
    )
    grid_search.fit(X_train, y_train)

    # Identifying the best model to choose
    columns = [f"param_{name}" for name in param_grid]
    columns += ["mean_test_score", "rank_test_score"]
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results["mean_test_score"] = -cv_results["mean_test_score"]
    cv_results[columns].sort_values(by="rank_test_score")

    params = cv_results[cv_results.rank_test_score == 1].iloc[0, 6]

    model = HistGradientBoostingClassifier(
        max_iter=params["max_iter"], max_depth=params["max_depth"]
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)

    return y_pred


def signal_strategy(
    series: pd.Series,
    ohlcv: Optional[pd.DataFrame] = None,
    ema_s: int = 15,
    ema_l: int = 45,
    roll_days: Optional[int] = 128,
    series_name: Optional[str] = "",
    model_type="Hurst",
):

    # series = Financial time Series
    # hurst_df is our Curst computed DataFrame
    # ema_s is our small moving average
    # ema_l is our large moving average
    # series_name is the name of our financial time series

    # Computing Moving averages
    ema_short = talib.EMA(series, timeperiod=ema_s).to_frame()
    ema_long = talib.EMA(series, timeperiod=ema_l).to_frame()

    if model_type == "ML":
        # Data for ML model strategy dataframe
        #model = train_model(ohlcv)
        #pred = model_predict(model, ohlcv)
         pred = train_model_predict(ohlcv=ohlcv)
         positions = take_positions_based_on_model_predictions(pred)
         signal_df = pd.DataFrame(
            positions,
            index=series.iloc[len(series) - len(pred) :].index,
            columns=[series_name],
        )
    else:
        # Data for Hurst model signal dataframe
        series_roll_prices = series.rolling(roll_days)
        series_rolling_hursts = series_roll_prices.apply(
            lambda s: hurst_exponent(s, method="RS")
        ).dropna()
        series_hurst_macd = get_macd(series_rolling_hursts)
        _, _, series_macd_signal = implement_macd_strategy(
            series.iloc[roll_days - 1 :], series_hurst_macd
        )
        signal_df = pd.DataFrame(
            series_macd_signal,
            index=series.iloc[roll_days - 1 :].index,
            columns=[series_name],
        )
    # signal_df.rename(columns={"signal":"bt_hsi"}, inplace=True)

    # Plotting Financial series with Signals and Moving Averages
    combo_df = bt.merge(signal_df, series, ema_short, ema_long)
    combo_df.columns = ["Signal Strategy", series_name, "EMA_short", "EMA_LONG"]
    ax_name = combo_df.columns[0]

    # Applying Strategy
    # Creating DataFrame of Financial Time Series
    df = pd.DataFrame(series)
    df.rename(columns={"Close": series_name}, inplace=True)

    # Running Strategy Strategy
    bt_strategy_cross = bt.Strategy(
        f"EMA_crossover with {model_type} model for " + series_name,
        [bt.algos.WeighTarget(signal_df), bt.algos.Rebalance()],
    )
    bt_backtest_cross = bt.Backtest(bt_strategy_cross, df)
    bt_result_cross = bt.run(bt_backtest_cross)

    return bt_result_cross, combo_df, ax_name


def get_macd(rolling_hurst: pd.Series, slow: int = 26, fast: int = 12, smooth: int = 9):
    exp1 = rolling_hurst.ewm(span=fast, adjust=False).mean()
    exp2 = rolling_hurst.ewm(span=slow, adjust=False).mean()
    diff_series = exp1 - exp2
    # mh_signal = pd.DataFrame(rolling_hurst).rename(columns={"Close": "signal"})
    macd = pd.DataFrame(diff_series).rename(columns={diff_series.name: "macd"})
    signal = pd.DataFrame(macd.ewm(span=smooth, adjust=False).mean()).rename(
        columns={"macd": "signal"}
    )

    hist = pd.DataFrame(macd["macd"] - signal["signal"]).rename(columns={0: "hist"})
    frames = [macd, signal, hist]
    # signal["signal"] += mh_signal["signal"]
    return pd.concat(frames, join="inner", axis=1)


def plot_macd(prices: pd.DataFrame, macd_df: pd.DataFrame, window: int = 128):
    ax1 = plt.subplot2grid((8, 1), (0, 0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((8, 1), (5, 0), rowspan=3, colspan=1)

    ax1.plot(prices.iloc[window:])
    ax1.set_ylabel("Closing prices")
    ax2.plot(macd_df["macd"], color="grey", linewidth=1.5, label="MACD")
    ax2.plot(macd_df["signal"], color="skyblue", linewidth=1.5, label="SIGNAL")

    for i in range(len(prices.iloc[window:])):
        if str(macd_df["hist"][i])[0] == "-":
            ax2.bar(prices.iloc[window:].index[i], macd_df["hist"][i], color="#ef5350")
        else:
            ax2.bar(prices.iloc[window:].index[i], macd_df["hist"][i], color="#26a69a")

    plt.legend(loc="lower right")
    plt.suptitle(f"{prices.columns[0]} closing prices, MACD and tradign signal")


def implement_macd_strategy_v2(prices: pd.Series, macd_df: pd.DataFrame):
    buy_price = []
    sell_price = []
    macd_signal = []
    signal = 0

    for i in range(len(macd_df)):
        if macd_df["macd"][i] > 0.5:
            if signal != 1:
                buy_price.append(prices[i])
                signal = 1
                macd_signal.append(signal)
            else:
                buy_price.append(np.nan)
                macd_signal.append(0)
            sell_price.append(np.nan)
        elif macd_df["macd"][i] < 0.5:
            if signal != -1:
                sell_price.append(prices[i])
                signal = -1
                macd_signal.append(signal)
            else:
                sell_price.append(np.nan)
                macd_signal.append(0)
            buy_price.append(np.nan)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            macd_signal.append(0)

    return buy_price, sell_price, macd_signal


def implement_macd_strategy(prices: pd.Series, macd_df: pd.DataFrame):
    buy_price = []
    sell_price = []
    macd_signal = []
    signal = 0

    for i in range(len(macd_df)):
        if macd_df["macd"][i] > macd_df["signal"][i]:
            if signal != 1:
                buy_price.append(prices[i])
                signal = 1
                macd_signal.append(signal)
            else:
                buy_price.append(np.nan)
                macd_signal.append(0)
            sell_price.append(np.nan)
        elif macd_df["macd"][i] < macd_df["signal"][i]:
            if signal != -1:
                sell_price.append(prices[i])
                signal = -1
                macd_signal.append(signal)
            else:
                sell_price.append(np.nan)
                macd_signal.append(0)
            buy_price.append(np.nan)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            macd_signal.append(0)

    return buy_price, sell_price, macd_signal


def plot_macd_strategy(
    series: pd.Series,
    asset_macd: pd.DataFrame,
    buy_price: list,
    sell_price: list,
    ticker: str,
):
    ax1 = plt.subplot2grid((8, 1), (0, 0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((8, 1), (5, 0), rowspan=3, colspan=1)

    ax1.plot(series, color="skyblue", linewidth=2, label=f"{ticker}")
    ax1.plot(
        series.index,
        buy_price,
        marker="^",
        color="green",
        markersize=10,
        label="BUY SIGNAL",
        linewidth=0,
    )
    ax1.plot(
        series.index,
        sell_price,
        marker="v",
        color="r",
        markersize=10,
        label="SELL SIGNAL",
        linewidth=0,
    )
    ax1.legend()
    ax1.set_title(f"{ticker} MACD SIGNALS")
    ax2.plot(asset_macd["macd"], color="grey", linewidth=1.5, label="MACD")
    ax2.plot(asset_macd["signal"], color="skyblue", linewidth=1.5, label="SIGNAL")

    for i in range(len(asset_macd)):
        if str(asset_macd["hist"][i])[0] == "-":
            ax2.bar(asset_macd.index[i], asset_macd["hist"][i], color="#ef5350")
        else:
            ax2.bar(asset_macd.index[i], asset_macd["hist"][i], color="#26a69a")
    plt.legend(loc="lower right")
    plt.show()


def strategy_return_df(series: pd.Series, asset_macd: pd.DataFrame, macd_signal: list):
    position = []
    for item in macd_signal:
        if item > 1:
            position.append(0)
        else:
            position.append(1)
    for i in range(len(series)):
        if macd_signal[i] == 1:
            position[i] = 1
        elif macd_signal[i] == -1:
            position[i] = 0
        else:
            position[i] = position[i - 1]
    macd = asset_macd["macd"]
    signal = asset_macd["signal"]
    macd_signal = (
        pd.DataFrame(macd_signal)
        .rename(columns={0: "macd_signal"})
        .set_index(series.index)
    )
    position = (
        pd.DataFrame(position)
        .rename(columns={0: "macd_position"})
        .set_index(series.index)
    )

    frames = [series, macd, signal, macd_signal, position]
    strategy = pd.concat(frames, join="inner", axis=1)

    asset_ret = pd.DataFrame(np.diff(series)).rename(columns={0: "returns"})
    macd_strategy_ret = []

    for i in range(len(asset_ret)):
        with contextlib.suppress(Exception):
            returns = asset_ret["returns"][i] * strategy["macd_position"][i]
            macd_strategy_ret.append(returns)
    return pd.DataFrame(macd_strategy_ret).rename(columns={0: "macd_returns"})


def strategy_profit(
    investment_value: int, series: pd.Series, asset_macd, macd_signal,
):
    macd_strategy_ret_df = strategy_return_df(series, asset_macd, macd_signal)
    number_of_stocks = floor(investment_value / series[0])
    macd_investment_ret = []

    for i in range(len(macd_strategy_ret_df["macd_returns"])):
        returns = number_of_stocks * macd_strategy_ret_df["macd_returns"][i]
        macd_investment_ret.append(returns)

    macd_investment_ret_df = pd.DataFrame(macd_investment_ret).rename(
        columns={0: "investment_returns"}
    )
    total_investment_ret = round(sum(macd_investment_ret_df["investment_returns"]), 2)
    profit_percentage = floor((total_investment_ret / investment_value) * 100)
    print(
        cl(
            f"Profit gained from the MACD strategy by investing $100k in asset : {total_investment_ret}",
            attrs=["bold"],
        )
    )
    print(
        cl(
            f"Profit percentage of the MACD strategy : {profit_percentage}%",
            attrs=["bold"],
        )
    )


def get_benchmark_performance(index: pd.Series, start_date: str, investment_value):

    benchmark = pd.DataFrame(np.diff(index)).rename(columns={0: "benchmark_returns"})

    number_of_stocks = floor(investment_value / index[0])
    benchmark_investment_ret = []
    for i in range(len(benchmark["benchmark_returns"])):
        returns = number_of_stocks * benchmark["benchmark_returns"][i]
        benchmark_investment_ret.append(returns)

    benchmark_return = pd.DataFrame(benchmark_investment_ret).rename(
        columns={0: "investment_returns"}
    )
    total_benchmark_investment_ret = round(
        sum(benchmark_return["investment_returns"]), 2
    )
    benchmark_profit_percentage = floor(
        (total_benchmark_investment_ret / investment_value) * 100
    )
    print(
        cl(
            f"Benchmark profit by investing $100k : {total_benchmark_investment_ret}",
            attrs=["bold"],
        )
    )
    print(
        cl(
            f"Benchmark Profit percentage : {benchmark_profit_percentage}%",
            attrs=["bold"],
        )
    )


def hurst_dsod(x):
    """Estimate Hurst exponent on data timeseries.

    The estimation is based on the discrete second order derivative. Consists on
    get two different noise of the original series and calculate the standard
    deviation and calculate the slope of two point with that values.
    source: https://gist.github.com/wmvanvliet/d883c3fe1402c7ced6fc

    Parameters
    ----------
    x : numpy array
        time series to estimate the Hurst exponent for.

    Returns
    -------
    h : float
        The estimation of the Hurst exponent for the given time series.

    References
    ----------
    Istas, J.; G. Lang (1994), “Quadratic variations and estimation of the local
    Hölder index of data Gaussian process,” Ann. Inst. Poincaré, 33, pp. 407–436.


    Notes
    -----
    This hurst_ets is data literal traduction of wfbmesti.m of waveleet toolbox
    from matlab.
    """
    y = np.cumsum(np.diff(x, axis=0), axis=0)

    # second order derivative
    b1 = [1, -2, 1]
    y1 = signal.lfilter(b1, 1, y, axis=0)
    y1 = y1[len(b1) - 1 :]  # first values contain filter artifacts

    # wider second order derivative
    b2 = [1, 0, -2, 0, 1]
    y2 = signal.lfilter(b2, 1, y, axis=0)
    y2 = y2[len(b2) - 1 :]  # first values contain filter artifacts

    s1 = np.mean(y1 ** 2, axis=0)
    s2 = np.mean(y2 ** 2, axis=0)

    return 0.5 * np.log2(s2 / s1)


def hurst_rs(x, min_chunksize, max_chunksize, num_chunksize):
    """Estimate the Hurst exponent using R/S method.

    Estimates the Hurst (H) exponent using the R/S method from the time series.
    The R/S method consists of dividing the series into pieces of equal size
    `series_len` and calculating the rescaled range. This repeats the process
    for several `series_len` values and adjusts data regression to obtain the H.
    `series_len` will take values between `min_chunksize` and `max_chunksize`,
    the step size from `min_chunksize` to `max_chunksize` can be controlled
    through the parameter `step_chunksize`.

    Parameters
    ----------
    x : 1D-array
        A time series to calculate hurst exponent, must have more elements
        than `min_chunksize` and `max_chunksize`.
    min_chunksize : int
        This parameter allow you control the minimum window size.
    max_chunksize : int
        This parameter allow you control the maximum window size.
    num_chunksize : int
        This parameter allow you control the size of the step from minimum to
        maximum window size. Bigger step means fewer calculations.
    out : 1-element-array, optional
        one element array to store the output.

    Returns
    -------
    H : float
        A estimation of Hurst exponent.

    References
    ----------
    Hurst, H. E. (1951). Long term storage capacity of reservoirs. ASCE
    Transactions, 116(776), 770-808.
    Alessio, E., Carbone, A., Castelli, G. et al. Eur. Phys. J. B (2002) 27:
    197. http://dx.doi.org/10.1140/epjb/e20020150
    """
    N = len(x)
    max_chunksize += 1
    rs_tmp = np.empty(N, dtype=np.float64)
    chunk_size_list = np.linspace(min_chunksize, max_chunksize, num_chunksize).astype(
        np.int64
    )
    rs_values_list = np.empty(num_chunksize, dtype=np.float64)

    # 1. The series is divided into chunks of chunk_size_list size
    for i in range(num_chunksize):
        chunk_size = chunk_size_list[i]

        # 2. it iterates on the indices of the first observation of each chunk
        number_of_chunks = int(len(x) / chunk_size)

        for idx in range(number_of_chunks):
            # next means no overlapping
            # convert index to index selection of each chunk
            ini = idx * chunk_size
            end = ini + chunk_size
            chunk = x[ini:end]

            # 2.1 Calculate the RS (chunk_size)
            z = np.cumsum(chunk - np.mean(chunk))
            rs_tmp[idx] = np.divide(
                np.max(z) - np.min(z), np.nanstd(chunk)  # range  # standar deviation
            )

        # 3. Average of RS(chunk_size)
        rs_values_list[i] = np.nanmean(rs_tmp[: idx + 1])

    # 4. calculate the Hurst exponent.
    H, c = np.linalg.lstsq(
        a=np.vstack((np.log(chunk_size_list), np.ones(num_chunksize))).T,
        b=np.log(rs_values_list),
        rcond=None,
    )[0]

    return H


def hurst_dma(prices, min_chunksize=8, max_chunksize=200, num_chunksize=5):
    """Estimate the Hurst exponent using R/S method.

    Estimates the Hurst (H) exponent using the DMA method from the time series.
    The DMA method consists on calculate the moving average of size `series_len`
    and subtract it to the original series and calculating the standard
    deviation of that result. This repeats the process for several `series_len`
    values and adjusts data regression to obtain the H. `series_len` will take
    values between `min_chunksize` and `max_chunksize`, the step size from
    `min_chunksize` to `max_chunksize` can be controlled through the parameter
    `step_chunksize`.

    Parameters
    ----------
    prices
    min_chunksize
    max_chunksize
    num_chunksize

    Returns
    -------
    hurst_exponent : float
        Estimation of hurst exponent.

    References
    ----------
    Alessio, E., Carbone, A., Castelli, G. et al. Eur. Phys. J. B (2002) 27:
    197. http://dx.doi.org/10.1140/epjb/e20020150

    """
    max_chunksize += 1
    N = len(prices)
    n_list = np.arange(min_chunksize, max_chunksize, num_chunksize, dtype=np.int64)
    dma_list = np.empty(len(n_list))
    factor = 1 / (N - max_chunksize)
    # sweeping n_list
    for i, n in enumerate(n_list):
        b = np.divide([n - 1] + (n - 1) * [-1], n)  # do the same as:  y - y_ma_n
        noise = np.power(signal.lfilter(b, 1, prices)[max_chunksize:], 2)
        dma_list[i] = np.sqrt(factor * np.sum(noise))

    H, const = np.linalg.lstsq(
        a=np.vstack([np.log10(n_list), np.ones(len(n_list))]).T,
        b=np.log10(dma_list),
        rcond=None,
    )[0]
    return H


def hurst_exponent(
    prices, min_chunksize=8, max_chunksize=200, num_chunksize=5, method="RS"
):
    """Estimates Hurst Exponent.

    Estimate the hurst exponent following one of 3 methods. Each method

    Parameters
    ----------
    prices : numpy.ndarray, pandas.Series or pandas.DataFrame
        A time series to estimate hurst exponent.
    min_chunksize : int, optional
        Minimum chunk  size of the original series. This parameter doesn't have
        any effect with DSOD method.
    max_chunksize : int, optional
        Maximum chunk size of the original series. This parameter doesn't have
        any effect with DSOD method.
    step_chunksize : int, optional
        Step used to select next the chunk size which divide the original
        series. This parameter doesn't have any effect with DSOD method.
    method : {'RS', 'DMA', 'DSOD', 'all'}
        The methods can take one of that values,
            RS : rescaled range.
            DMA : deviation moving average.
            DSOD : discrete second order derivative.


    Returns
    -------
    hurst_exponent : float
        Estimation of hurst_exponent according to the method selected.

    References
    ----------
    RS : Hurst, H. E. (1951). Long term storage capacity of reservoirs. ASCE
         Transactions, 116(776), 770-808.
    DMA : Alessio, E., Carbone, A., Castelli, G. et al. Eur. Phys. J. B (2002)
         27: 197. http://dx.doi.org/10.1140/epjb/e20020150
    DSOD : Istas, J.; G. Lang (1994), “Quadratic variations and estimation of
        the local Hölder index of data Gaussian process,” Ann. Inst. Poincaré,
        33, pp. 407–436.

    Notes
    -----
    The hurst exponent is an estimation which is important because there is no
    data closed equation for it instead we have some methods to estimate it with
    high variations among them.

    See Also
    --------
    hurst_rs, hurst_dma, hurst_dsod
    """
    if len(prices) == 0:
        return np.nan
    # extract array
    arr = prices.__array__()
    # choose data method
    if method == "RS":
        if prices.ndim > 1:
            h = hurst_rs(
                np.diff(arr, axis=0).T, min_chunksize, max_chunksize, num_chunksize
            )
        else:
            h = hurst_rs(np.diff(arr), min_chunksize, max_chunksize, num_chunksize)
    elif method == "DMA":
        h = hurst_dma(arr, min_chunksize, max_chunksize, num_chunksize)
    elif method == "DSOD":
        h = hurst_dsod(arr)
    elif method == "all":
        return [
            hurst_exponent(arr, min_chunksize, max_chunksize, num_chunksize, "RS"),
            hurst_exponent(arr, min_chunksize, max_chunksize, num_chunksize, "DMA"),
            hurst_exponent(arr, min_chunksize, max_chunksize, num_chunksize, "DSOD"),
        ]
    else:
        raise NotImplementedError("The method choose is not implemented.")

    return h
