# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
from warnings import simplefilter
import numpy as np  # noqa
import pandas as pd  # noqa
import math
from pandas import DataFrame
from functools import reduce
from typing import Optional
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter, stoploss_from_absolute)
from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade
from datetime import datetime
# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
import freqtrade.vendor.qtpylib.indicators as qtpylib

import warnings
warnings.filterwarnings(
    'ignore', message='The objective has been evaluated at this point before.')
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.options.mode.chained_assignment = None

# This class is a sample. Feel free to customize it.

class EMA3SRSI(IStrategy):

    def custom_stochRSI_TravingView_Style(self, close, length=14, rsi_length=14, k=3, d=3):
        # Results between 0 and 1
        """Indicator: Stochastic RSI Oscillator (STOCHRSI)
        Should be similar to TradingView's calculation"""
        if k < 0:
            raise Exception("k cannot be negative")
        if d < 0:
            raise Exception("d cannot be negative")
        # Calculate Result
        rsi_ = pta.rsi(close, length=rsi_length, talib=False)
        lowest_rsi = rsi_.rolling(length).min()
        highest_rsi = rsi_.rolling(length).max()
        stochrsi = 100.0 * (rsi_ - lowest_rsi) / pta.non_zero_range(highest_rsi, lowest_rsi)
        if k > 0:
            stochrsi_k = pta.ma('sma', stochrsi, length=k, talib=False)
            stochrsi_d = pta.ma('sma', stochrsi_k, length=d, talib=False)
        else:
            stochrsi_k = None
            stochrsi_d = None
        return (stochrsi/100.0).round(4), (stochrsi_k/100.0).round(4), (stochrsi_d/100.0).round(4)

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    stochWindow = 14
    rsi_length = 14
    atr_per = 14

    position_adjustment_enable = False

    stochOverSold = DecimalParameter(0.1, 0.9, decimals=1, default=0.1, space="buy", optimize=True)
    #stochOverBought = 0.8

    UP = IntParameter(3, 15, default=12, space="buy", optimize=True)
    DOWN = IntParameter(3, 15, default=10, space="buy", optimize=True)

    ema1 = IntParameter(3, 500, default=32, space="buy", optimize=True)
    ema2 = IntParameter(3, 500, default=177, space="buy", optimize=True)
    ema3 = IntParameter(3, 598 , default=315, space="buy", optimize=True)

    StochRSI_direction = CategoricalParameter(["upper_lower", "lower_upper"], default="upper_lower", space="buy", optimize=True)

    # Can this strategy go short?
    can_short: bool = False
    use_custom_stoploss: bool = True

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 0.15,
        "2880": -1
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.95

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Optimal timeframe for the strategy.
    timeframe = '5m'

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 599

    # Optional order type mapping.
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'limit',
        'stoploss_on_exchange': True
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    ## DCA parameters
    initial_safety_order_trigger = DecimalParameter(-0.10, -0.01, decimals=2, default=-0.05, space="buy", optimize=True)
    max_safety_orders = IntParameter(0, 3, default=0, space="buy", optimize=True)
    safety_order_step_scale = DecimalParameter(0.10, 2.00, decimals=1, default=1.00, space="buy", optimize=True)
    safety_order_volume_scale = DecimalParameter(0.10, 2.00, decimals=1, default=1.00, space="buy", optimize=True)

    if max_safety_orders.value >= 1:
        position_adjustment_enable = True
    else:
        position_adjustment_enable = False

    # max_dca_multiplier calculation
    max_dca_multiplier = (1.0 + float(max_safety_orders.value))
    if (max_safety_orders.value > 0):
        if (safety_order_volume_scale.value > 1.0):
            max_dca_multiplier = (2.0 + (safety_order_volume_scale.value * (math.pow(safety_order_volume_scale.value, (float(max_safety_orders.value) - 1.0)) - 1.0) / (safety_order_volume_scale.value - 1.0)))
        elif (safety_order_volume_scale.value < 1.0):
            max_dca_multiplier = (2.0 + (safety_order_volume_scale.value * (1.0 - math.pow(safety_order_volume_scale.value, (float(max_safety_orders.value) - 1.0))) / (1.0 - safety_order_volume_scale.value)))

    def informative_pairs(self):
        """
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        """
        dataframe['EMA1'] = ta.EMA(dataframe, timeperiod=int(self.ema1.value))
        dataframe['EMA2'] = ta.EMA(dataframe, timeperiod=int(self.ema2.value))
        dataframe['EMA3'] = ta.EMA(dataframe, timeperiod=int(self.ema3.value))
        
        dataframe['ema1_val']=self.ema1.value
        dataframe['ema2_val']=self.ema2.value
        dataframe['ema3_val']=self.ema3.value
        
        _, dataframe['K'], dataframe['D'] = self.custom_stochRSI_TravingView_Style(close=dataframe['close'], length=self.stochWindow, rsi_length=self.rsi_length, k=3, d=3)
        
        dataframe['ATR'] = ta.ATR(dataframe, timeperiod=self.atr_per)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        """
        conditions = []
        conditions.append(dataframe['ema2_val'] > dataframe['ema1_val'])
        conditions.append(dataframe['ema3_val'] > dataframe['ema2_val'])
        conditions.append(dataframe['EMA1'] >= dataframe['EMA2'])
        conditions.append(dataframe['EMA2'] >= dataframe['EMA3'])
        conditions.append(dataframe['close'] >= dataframe['EMA1'])
        conditions.append(dataframe['K'] < self.stochOverSold.value)
        conditions.append(dataframe['D'] < self.stochOverSold.value)

        if self.StochRSI_direction.value == "upper_lower":
            conditions.append(dataframe['K'].shift(1) > dataframe['D'].shift(1))
            conditions.append(dataframe['K'] <= dataframe['D'])
        elif self.StochRSI_direction.value == "lower_upper":
            conditions.append(dataframe['K'].shift(1) < dataframe['D'].shift(1))
            conditions.append(dataframe['K'] >= dataframe['D'])

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        """
        dataframe.loc[:, 'exit_long'] = 0
        return dataframe

    def get_average_entry_price(self, trade: Trade):
        filled_entries = trade.select_filled_orders(trade.entry_side)
        nb = float(trade.nr_of_successful_entries)
        stake_amount = 0.0
        for ff in filled_entries:
            stake_amount = stake_amount + float(ff.cost)
        return stake_amount/nb
        
    # USED FOR STOP LOSS
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        """
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        #last_candle = dataframe.iloc[-1].squeeze()
        trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        # Look up trade candle.
        trade_candle = dataframe.loc[dataframe['date'] == trade_date]
        # trade_candle may be empty for trades that just opened as it is still incomplete.
        if not trade_candle.empty:
            trade_candle = trade_candle.squeeze()
            # avg = self.get_average_entry_price(trade)
            c2 = current_rate < trade.open_rate - trade_candle['ATR']*self.DOWN.value
            if c2:
                return 'stop_loss'

    # USED AS A TAKE PROFIT
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze() # = current candle
        trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        # Look up trade candle.
        trade_candle = dataframe.loc[dataframe['date'] == trade_date]
        # trade_candle may be empty for trades that just opened as it is still incomplete.
        if not trade_candle.empty:
            trade_candle = trade_candle.squeeze()
            # avg = self.get_average_entry_price(trade)
            c1 = current_rate > trade.open_rate + trade_candle['ATR']*self.UP.value
            if c1:
                return -0.0002
        return -0.95

    # Let unlimited stakes leave funds open for DCA orders
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            **kwargs) -> float:
        """
        """
        if self.config['stake_amount'] == 'unlimited':
            return proposed_stake / self.max_dca_multiplier
        else:
            return proposed_stake

    # DCA
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):
        """
        """
        if current_profit > self.initial_safety_order_trigger.value:
            return None
        if self.max_safety_orders.value == 0:
            return None

        count_of_buys = trade.nr_of_successful_entries

        if 1 <= count_of_buys <= self.max_safety_orders.value:
            safety_order_trigger = (abs(self.initial_safety_order_trigger.value) * count_of_buys)
            if (self.safety_order_step_scale.value > 1):
                safety_order_trigger = abs(self.initial_safety_order_trigger.value) + (abs(self.initial_safety_order_trigger.value) * self.safety_order_step_scale.value * (math.pow(self.safety_order_step_scale.value,(count_of_buys - 1)) - 1) / (self.safety_order_step_scale.value - 1))
            elif (self.safety_order_step_scale.value < 1):
                safety_order_trigger = abs(self.initial_safety_order_trigger.value) + (abs(self.initial_safety_order_trigger.value) * self.safety_order_step_scale.value * (1 - math.pow(self.safety_order_step_scale.value,(count_of_buys - 1))) / (1 - self.safety_order_step_scale.value))

            if current_profit <= (-1.0* abs(safety_order_trigger)):
                try:
                    stake_amount = self.wallets.get_trade_stake_amount(trade.pair, None)
                    # This calculates base order size
                    stake_amount = stake_amount / self.max_dca_multiplier
                    # This then calculates current safety order size
                    stake_amount = stake_amount * math.pow(self.safety_order_volume_scale.value, (count_of_buys - 1))
                    amount = stake_amount / current_rate
                    #print(f"Initiating safety order buy #{count_of_buys} for {trade.pair} with stake amount of {stake_amount} which equals {amount}")
                    return stake_amount
                except Exception as exception:
                    #print(f'Error occured while trying to get stake amount for {trade.pair}: {str(exception)}')
                    return None
        return None
