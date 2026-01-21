"""
Shared utility functions for Data Driven Investing notebooks.

This module contains commonly used functions across multiple notebooks
to reduce code duplication and keep teaching cells focused on concepts.

Usage:
    from utils import get_factors, get_daily_wrds, MVE, Diagnostics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# =============================================================================
# Data Loading Functions
# =============================================================================

def get_factors(factors='CAPM', freq='daily'):
    """
    Load Fama-French factor data from Ken French's data library.
    
    Parameters
    ----------
    factors : str
        Factor model to load. Options: 'CAPM', 'FF3', 'FF5', 'ff6'
    freq : str
        Data frequency. Options: 'daily', 'monthly'
    
    Returns
    -------
    pd.DataFrame
        DataFrame with factor returns (already divided by 100)
    
    Example
    -------
    >>> df = get_factors('FF3', freq='monthly')
    >>> df.head()
    """
    import pandas_datareader.data as web
    
    if freq == 'monthly':
        freq_label = ''
    else:
        freq_label = '_' + freq

    if factors == 'CAPM':
        fama_french = web.DataReader("F-F_Research_Data_Factors" + freq_label, 
                                      "famafrench", start="1921-01-01")
        daily_data = fama_french[0]
        df_factor = daily_data[['RF', 'Mkt-RF']]
        
    elif factors == 'FF3':
        fama_french = web.DataReader("F-F_Research_Data_Factors" + freq_label, 
                                      "famafrench", start="1921-01-01")
        daily_data = fama_french[0]
        df_factor = daily_data[['RF', 'Mkt-RF', 'SMB', 'HML']]
        
    elif factors == 'FF5':
        fama_french = web.DataReader("F-F_Research_Data_Factors" + freq_label, 
                                      "famafrench", start="1921-01-01")
        daily_data = fama_french[0]
        df_factor = daily_data[['RF', 'Mkt-RF', 'SMB', 'HML']]
        
        fama_french2 = web.DataReader("F-F_Research_Data_5_Factors_2x3" + freq_label, 
                                       "famafrench", start="1921-01-01")
        daily_data2 = fama_french2[0]
        df_factor2 = daily_data2[['RMW', 'CMA']]
        df_factor = df_factor.merge(df_factor2, on='Date', how='outer')
        
    else:  # ff6 or default
        fama_french = web.DataReader("F-F_Research_Data_Factors" + freq_label, 
                                      "famafrench", start="1921-01-01")
        daily_data = fama_french[0]
        df_factor = daily_data[['RF', 'Mkt-RF', 'SMB', 'HML']]
        
        fama_french2 = web.DataReader("F-F_Research_Data_5_Factors_2x3" + freq_label, 
                                       "famafrench", start="1921-01-01")
        daily_data2 = fama_french2[0]
        df_factor2 = daily_data2[['RMW', 'CMA']]
        df_factor = df_factor.merge(df_factor2, on='Date', how='outer')
        
        fama_french = web.DataReader("F-F_Momentum_Factor" + freq_label, 
                                      "famafrench", start="1921-01-01")
        df_factor = df_factor.merge(fama_french[0], on='Date')
        df_factor.columns = ['RF', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
    
    if freq == 'monthly':
        df_factor.index = pd.to_datetime(df_factor.index.to_timestamp())
    else:
        df_factor.index = pd.to_datetime(df_factor.index)

    return df_factor / 100


def get_daily_wrds_single_ticker(ticker, conn, dividends=True):
    """
    Get daily price and dividend data for a single ticker from WRDS.
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol
    conn : wrds.Connection
        Active WRDS database connection
    dividends : bool
        If True, return Price and Dividend columns. If False, return returns.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with date index and P, D columns (or ret column)
    """
    tickers = [ticker]
    
    # Retrieve PERMNOs for the specified tickers
    permnos = conn.get_table(library='crsp', table='stocknames', 
                              columns=['permno', 'ticker', 'namedt', 'nameenddt'])
    permnos['nameenddt'] = pd.to_datetime(permnos['nameenddt'])
    permnos = permnos[(permnos['ticker'].isin(tickers)) & 
                      (permnos['nameenddt'] == permnos['nameenddt'].max())]
    
    permno_list = [permnos['permno'].unique().tolist()[0]]
    print(f"Loading data for PERMNO: {permno_list}")

    query = f"""
        SELECT permno, date, ret, retx, prc       
        FROM crsp.dsf
        WHERE permno IN ({','.join(map(str, permno_list))})
        ORDER BY date
    """
    daily_returns = conn.raw_sql(query, date_cols=['date'])
    daily_returns = daily_returns.merge(permnos[['permno', 'ticker']], 
                                         on='permno', how='left')
    
    if dividends:
        daily_returns['D'] = ((daily_returns.ret - daily_returns.retx) * 
                              daily_returns.prc.abs().shift(1))
        daily_returns['P'] = daily_returns.prc.abs()
        daily_returns = daily_returns[['date', 'P', 'D']].set_index('date').dropna()
    else:
        daily_returns = daily_returns[['date', 'ret']].set_index('date').dropna()

    return daily_returns


def get_daily_wrds(conn, tickers=None):
    """
    Get daily returns for multiple tickers from WRDS.
    
    Parameters
    ----------
    conn : wrds.Connection
        Active WRDS database connection
    tickers : list
        List of stock ticker symbols
    
    Returns
    -------
    pd.DataFrame
        DataFrame with date index and ticker columns containing returns
    """
    permnos = conn.get_table(library='crsp', table='stocknames', 
                              columns=['permno', 'ticker', 'namedt', 'nameenddt'])
    permnos['nameenddt'] = pd.to_datetime(permnos['nameenddt'])
    permnos = permnos[(permnos['ticker'].isin(tickers)) & 
                      (permnos['nameenddt'] == permnos['nameenddt'].max())]
    
    permno_list = permnos['permno'].unique().tolist()
    print(f"Loading data for PERMNOs: {permno_list}")

    query = f"""
        SELECT permno, date, ret, retx, prc       
        FROM crsp.dsf
        WHERE permno IN ({','.join(map(str, permno_list))})
        ORDER BY date, permno
    """
    daily_returns = conn.raw_sql(query, date_cols=['date'])
    daily_returns = daily_returns.merge(permnos[['permno', 'ticker']], 
                                         on='permno', how='left')
    daily_returns = daily_returns.pivot(index='date', columns='ticker', values='ret')
    daily_returns = daily_returns[tickers]

    return daily_returns


# =============================================================================
# Portfolio Construction Functions
# =============================================================================

def MVE(df, VolTarget=0.1/12**0.5):
    """
    Compute Mean-Variance Efficient portfolio weights.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of asset returns (excess returns recommended)
    VolTarget : float
        Target portfolio volatility (default: 10% annualized for monthly data)
    
    Returns
    -------
    np.array
        Portfolio weights scaled to target volatility
    """
    VarR = df.cov()
    ER = df.mean()
    W = ER @ np.linalg.inv(VarR)
    VarW = W @ VarR @ W
    w = VolTarget / VarW**0.5
    Ww = w * W
    return Ww


# =============================================================================
# Performance Evaluation Functions
# =============================================================================

def Diagnostics(W, df, R=None):
    """
    Compute comprehensive diagnostics for a portfolio strategy.
    
    Parameters
    ----------
    W : array-like
        Portfolio weights (or 0 if passing R directly)
    df : pd.DataFrame
        DataFrame containing 'RF', 'Mkt-RF', and asset returns
    R : pd.Series, optional
        Pre-computed portfolio returns (if W=0)
    
    Returns
    -------
    pd.DataFrame
        Single-column DataFrame with diagnostic metrics
    """
    results = {}
    
    Rf = df['RF']
    Factor = df['Mkt-RF']
    df_assets = df.drop(columns=['RF'])
    
    if R is None:
        R = df_assets @ W
    
    T = R.shape[0]
    
    # Basic statistics
    results['SR'] = R.mean() / R.std() * 12**0.5
    results['SR_factor'] = Factor.mean() / Factor.std() * 12**0.5
    results['Vol'] = R.std() * 12**0.5
    results['Vol_factor'] = Factor.std() * 12**0.5
    results['mean'] = R.mean() * 12
    results['t_mean'] = R.mean() / R.std() * T**0.5
    results['mean_factor'] = Factor.mean() * 12
    results['t_mean_factor'] = Factor.mean() / Factor.std() * T**0.5
    
    # Alpha regression
    x = sm.add_constant(Factor)
    y = R
    regresult = sm.OLS(y, x).fit()
    results['alpha'] = regresult.params[0] * 12
    results['t_alpha'] = regresult.tvalues[0]
    results['AR'] = results['alpha'] / (regresult.resid.std() * 12**0.5)
    
    # Tail statistics
    results['tails'] = ((R < -3*R.std()).mean() + (R > 3*R.std()).mean())
    results['tails_factor'] = ((Factor < -3*Factor.std()).mean() + 
                               (Factor > 3*Factor.std()).mean())
    results['min_ret'] = R.min()
    results['min_factor'] = Factor.min()
    
    # Plot cumulative returns
    fig, ax = plt.subplots(1, figsize=(10, 5))
    (R + Rf + 1).cumprod().plot(logy=True, label='Portfolio')
    (Rf + Factor + 1).cumprod().plot(logy=True, label='Market')
    plt.legend()
    plt.title('Cumulative Returns (Log Scale)')
    plt.ylabel('Growth of $1')
    plt.xlabel('Date')
    
    formatted_dict = {key: [value] for key, value in results.items()}
    return pd.DataFrame(formatted_dict).T


def SR_vol(R):
    """
    Compute standard error of Sharpe Ratio using normal assumption.
    
    Parameters
    ----------
    R : pd.Series
        Return series
    
    Returns
    -------
    float
        Standard error of Sharpe Ratio
    """
    SR = R.mean() / R.std()
    T = R.shape[0]
    return (1 / (T - 1) * (1 + SR**2 / 2))**0.5


def SR_vol_boot(R, N=10000):
    """
    Compute standard error of Sharpe Ratio using bootstrap.
    
    Parameters
    ----------
    R : pd.Series
        Return series
    N : int
        Number of bootstrap samples
    
    Returns
    -------
    tuple
        (standard deviation, 5th percentile)
    """
    SR = np.array([])
    T = R.shape[0]
    for i in range(N):
        Rs = R.sample(n=T, replace=True)
        SR = np.append(SR, Rs.mean() / Rs.std())
    return Rs.std(), np.percentile(SR, 5)


# =============================================================================
# Plotting Utilities
# =============================================================================

def set_plot_style():
    """Set consistent matplotlib style for all notebooks."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = [10, 6]
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12


def plot_drawdown(returns, rf=None, title='Drawdown Analysis'):
    """
    Plot cumulative performance and drawdown.
    
    Parameters
    ----------
    returns : pd.Series
        Excess return series
    rf : pd.Series, optional
        Risk-free rate series
    title : str
        Plot title
    """
    if rf is None:
        rf = pd.Series(0, index=returns.index)
    
    cumulative = (returns + rf + 1).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    cumulative.plot(ax=ax[0], logy=True, label='Cumulative')
    running_max.plot(ax=ax[0], logy=True, label='High Water Mark', alpha=0.7)
    ax[0].set_title('Cumulative Performance')
    ax[0].set_ylabel('Growth of $1 (log scale)')
    ax[0].legend()
    
    drawdown.plot(ax=ax[1], color='red', alpha=0.7)
    ax[1].fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
    ax[1].set_title('Drawdown from Peak')
    ax[1].set_ylabel('Drawdown (%)')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig
