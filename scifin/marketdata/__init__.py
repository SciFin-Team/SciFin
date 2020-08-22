# __init__.py
__version__ = "0.1.0"
__author__ = "Fabien Nugier"

"""
The :mod:`scifin.marketdata` module includes methods for market data gathering.
"""

#---------#---------#---------#---------#---------#---------#---------#---------#---------#


from .marketdata import get_sp500_tickers, get_assets_from_yahoo, get_marketcap_today, \
                        market_EWindex, market_CWindex

from .simuldata  import Market, \
                        set_market_names, create_market_returns, create_market_shares, plot_market_components, \
                        propagate_individual, evaluation_dates, \
                        find_tick_before_eval, limited_propagation, portfolio_vol, \
                        fitness_calculation, \
                        visualize_portfolios_1, visualize_portfolios_2, show_allocation_distrib, \
                        config_4n, plot_diff_GenPort_CW, plot_asset_evol

