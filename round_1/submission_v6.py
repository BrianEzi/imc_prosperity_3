from datamodel import Order, TradingState, OrderDepth
from typing import List
import jsonpickle
from config_hyper import HYPERPARAMS

# ===== Parameters =====

# General parameters
POSITION_LIMIT = 50

# Common late-day management
LATE_DAY_TIMESTAMP = HYPERPARAMS['LATE_DAY_TIMESTAMP']       # Timestamp threshold near market close
LATE_DAY_SIZE_FACTOR = HYPERPARAMS['LATE_DAY_SIZE_FACTOR']      # Order sizes scaled by this factor late in day
LATE_DAY_SPREAD_FACTOR = HYPERPARAMS['LATE_DAY_SPREAD_FACTOR']    # Spread (market-making edge) is widened by this factor late in day

# ------------------
# Rainforest Resin (fixed fair value)
# ------------------
RESIN_FAIR_VALUE = 10000
RESIN_TAKE_WIDTH = HYPERPARAMS['RESIN_TAKE_WIDTH']
RESIN_MM_EDGE = HYPERPARAMS['RESIN_MM_EDGE']

# ------------------
# Kelp (dynamic fair value using EMA)
# ------------------
KELP_TAKE_WIDTH = HYPERPARAMS['KELP_TAKE_WIDTH']
KELP_MM_EDGE = HYPERPARAMS['KELP_MM_EDGE']
KELP_BETA = HYPERPARAMS['KELP_BETA']   # Smoothing parameter for EMA update

# ------------------
# Squid Ink (dynamic fair value using trend adjustment)
# ------------------
SQUID_TAKE_WIDTH = HYPERPARAMS['SQUID_TAKE_WIDTH']
SQUID_MM_EDGE = HYPERPARAMS['SQUID_MM_EDGE']
SQUID_BETA = HYPERPARAMS['SQUID_BETA'] # Smoothing parameter for EMA update (EMA is used for fair value)
SQUID_TREND_THRESHOLD = HYPERPARAMS['SQUID_TREND_THRESHOLD']  # Minimum difference to consider as a trend signal
SQUID_TREND_BIAS = HYPERPARAMS['SQUID_TREND_BIAS']       # Bias factor: how much to shift fair value in a downtrend
SQUID_VOL_THRESHOLD = HYPERPARAMS['SQUID_VOL_THRESHOLD']     # Volume threshold for immediate order taking (if volume > threshold, skip)
# EMA windows for trend detection (squid ink)
SQUID_SHORT_EMA_WINDOW = HYPERPARAMS['SQUID_SHORT_EMA_WINDOW']
SQUID_LONG_EMA_WINDOW = HYPERPARAMS['SQUID_LONG_EMA_WINDOW']

# ===== End Parameters =====

class Trader:
    def __init__(self):
        self.product_resin = "RAINFOREST_RESIN"
        self.product_kelp = "KELP"
        self.product_squid = "SQUID_INK"
        # For state persistence, we will store dynamic variables for Kelp and Squid Ink.
        # (Rainforest Resin uses fixed fair value.)

    # ------------------
    # Utility: EMA Update Function
    # ------------------
    def update_ema(self, last_ema, price, window):
        # Standard EMA: alpha = 2/(window+1)
        alpha = 2 / (window + 1)
        if last_ema is None:
            return price
        return alpha * price + (1 - alpha) * last_ema

    # ------------------
    # Kelp Fair Value: dynamic via EMA smoothing.
    # ------------------
    def compute_kelp_fair_value(self, depth: OrderDepth, trader_state: dict) -> float:
        if not depth.buy_orders or not depth.sell_orders:
            return trader_state.get("kelp_last_fair", 10000)
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        observed_mid = (best_bid + best_ask) / 2
        last_fair = trader_state.get("kelp_last_fair", observed_mid)
        # EMA update with smoothing factor
        new_fair = last_fair + KELP_BETA * (observed_mid - last_fair)
        trader_state["kelp_last_fair"] = new_fair
        return new_fair

    # ------------------
    # Squid Ink Fair Value: dynamic update with trend adjustment.
    # Uses short-term and long-term EMAs.
    # ------------------
    def compute_squid_fair_value(self, depth: OrderDepth, trader_state: dict) -> float:
        if not depth.buy_orders or not depth.sell_orders:
            return trader_state.get("squid_last_fair", 5000)
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        observed_mid = (best_bid + best_ask) / 2

        # Update short-term EMA and long-term EMA:
        short_ema = trader_state.get("squid_short_ema", observed_mid)
        long_ema = trader_state.get("squid_long_ema", observed_mid)
        short_ema = self.update_ema(short_ema, observed_mid, SQUID_SHORT_EMA_WINDOW)
        long_ema = self.update_ema(long_ema, observed_mid, SQUID_LONG_EMA_WINDOW)
        trader_state["squid_short_ema"] = short_ema
        trader_state["squid_long_ema"] = long_ema

        # Compute baseline EMA fair value: standard update
        last_fair = trader_state.get("squid_last_fair", observed_mid)
        new_fair = (1 - SQUID_BETA) * last_fair + SQUID_BETA * observed_mid

        # Compute trend strength from EMAs:
        trend_strength = short_ema - long_ema  # Negative indicates downward trend.
        if trend_strength < -SQUID_TREND_THRESHOLD:
            # Bias fair value downward in a strong downtrend.
            new_fair += SQUID_TREND_BIAS * trend_strength  # trend_strength is negative.
        trader_state["squid_last_fair"] = new_fair
        trader_state["squid_last_mid"] = observed_mid
        return new_fair

    # ------------------
    # Squid Ink Volatility: simple measure by difference between current mid and last mid.
    # ------------------
    def compute_squid_volatility(self, depth: OrderDepth, trader_state: dict) -> float:
        if not depth.buy_orders or not depth.sell_orders:
            return 0
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        current_mid = (best_bid + best_ask) / 2
        last_mid = trader_state.get("squid_last_mid", current_mid)
        volatility = abs(current_mid - last_mid)
        trader_state["squid_last_mid"] = current_mid
        return volatility

    # ------------------
    # Late-day risk adjustment: A scaling factor based on timestamp.
    # ------------------
    def get_late_day_factor(self, timestamp: int) -> float:
        if timestamp >= LATE_DAY_TIMESTAMP:
            return LATE_DAY_SIZE_FACTOR
        else:
            return 1.0

    # ------------------
    # Core run() method: Process all assets and adjust orders based on trend and late-day factors.
    # ------------------
    def run(self, state: TradingState):
        # Load persistent trader state
        trader_state = {}
        if state.traderData and state.traderData != "":
            trader_state = jsonpickle.decode(state.traderData)

        result = {}
        late_day_factor = self.get_late_day_factor(state.timestamp)

        # --- Process Rainforest Resin ---
        resin = "RAINFOREST_RESIN"
        if resin in state.order_depths:
            resin_depth: OrderDepth = state.order_depths[resin]
            orders_resin: List[Order] = []
            current_position = state.position.get(resin, 0)
            # Use fixed fair value
            fair_value_resin = RESIN_FAIR_VALUE
            # Adjust effective parameters by late day factor: widen spread, reduce order size.
            effective_take_width_resin = RESIN_TAKE_WIDTH * (1 if late_day_factor == 1.0 else LATE_DAY_SPREAD_FACTOR)
            effective_mm_edge_resin = RESIN_MM_EDGE * (1 if late_day_factor == 1.0 else LATE_DAY_SPREAD_FACTOR)
            size_multiplier = late_day_factor

            if resin_depth.sell_orders:
                best_ask = min(resin_depth.sell_orders.keys())
                best_ask_volume = -resin_depth.sell_orders[best_ask]
                if best_ask <= fair_value_resin - effective_take_width_resin:
                    quantity = min(best_ask_volume, int((POSITION_LIMIT - current_position) * size_multiplier))
                    if quantity > 0:
                        orders_resin.append(Order(resin, best_ask, quantity))
            if resin_depth.buy_orders:
                best_bid = max(resin_depth.buy_orders.keys())
                best_bid_volume = resin_depth.buy_orders[best_bid]
                if best_bid >= fair_value_resin + effective_take_width_resin:
                    quantity = min(best_bid_volume, int((POSITION_LIMIT + current_position) * size_multiplier))
                    if quantity > 0:
                        orders_resin.append(Order(resin, best_bid, -quantity))
            if current_position < POSITION_LIMIT:
                buy_qty = int((POSITION_LIMIT - current_position) * size_multiplier)
                orders_resin.append(Order(resin, fair_value_resin - effective_mm_edge_resin, buy_qty))
            if current_position > -POSITION_LIMIT:
                sell_qty = int((POSITION_LIMIT + current_position) * size_multiplier)
                orders_resin.append(Order(resin, fair_value_resin + effective_mm_edge_resin, -sell_qty))
            result[resin] = orders_resin

        # --- Process Kelp ---
        kelp = "KELP"
        if kelp in state.order_depths:
            kelp_depth: OrderDepth = state.order_depths[kelp]
            orders_kelp: List[Order] = []
            current_position = state.position.get(kelp, 0)
            kelp_fair_value = self.compute_kelp_fair_value(kelp_depth, trader_state)
            effective_take_width_kelp = KELP_TAKE_WIDTH * (1 if late_day_factor == 1.0 else LATE_DAY_SPREAD_FACTOR)
            effective_mm_edge_kelp = KELP_MM_EDGE * (1 if late_day_factor == 1.0 else LATE_DAY_SPREAD_FACTOR)
            size_multiplier = late_day_factor

            if kelp_depth.sell_orders:
                best_ask = min(kelp_depth.sell_orders.keys())
                best_ask_volume = -kelp_depth.sell_orders[best_ask]
                if best_ask <= kelp_fair_value - effective_take_width_kelp:
                    quantity = min(best_ask_volume, int((POSITION_LIMIT - current_position) * size_multiplier))
                    if quantity > 0:
                        orders_kelp.append(Order(kelp, best_ask, quantity))
            if kelp_depth.buy_orders:
                best_bid = max(kelp_depth.buy_orders.keys())
                best_bid_volume = kelp_depth.buy_orders[best_bid]
                if best_bid >= kelp_fair_value + effective_take_width_kelp:
                    quantity = min(best_bid_volume, int((POSITION_LIMIT + current_position) * size_multiplier))
                    if quantity > 0:
                        orders_kelp.append(Order(kelp, best_bid, -quantity))
            if current_position < POSITION_LIMIT:
                buy_qty = int((POSITION_LIMIT - current_position) * size_multiplier)
                orders_kelp.append(Order(kelp, int(round(kelp_fair_value - effective_mm_edge_kelp)), buy_qty))
            if current_position > -POSITION_LIMIT:
                sell_qty = int((POSITION_LIMIT + current_position) * size_multiplier)
                orders_kelp.append(Order(kelp, int(round(kelp_fair_value + effective_mm_edge_kelp)), -sell_qty))
            result[kelp] = orders_kelp

        # --- Process Squid Ink ---
        squid = "SQUID_INK"
        if squid in state.order_depths:
            squid_depth: OrderDepth = state.order_depths[squid]
            orders_squid: List[Order] = []
            current_position = state.position.get(squid, 0)
            squid_fair_value = self.compute_squid_fair_value(squid_depth, trader_state)
            squid_volatility = self.compute_squid_volatility(squid_depth, trader_state)
            effective_take_width_squid = SQUID_TAKE_WIDTH * (1 if late_day_factor == 1.0 else LATE_DAY_SPREAD_FACTOR)
            effective_mm_edge_squid = SQUID_MM_EDGE * (1 if late_day_factor == 1.0 else LATE_DAY_SPREAD_FACTOR)
            size_multiplier = late_day_factor

            # Order taking: use volume filtering as before.
            if squid_depth.sell_orders:
                best_ask = min(squid_depth.sell_orders.keys())
                best_ask_volume = -squid_depth.sell_orders[best_ask]
                if best_ask <= squid_fair_value - effective_take_width_squid and best_ask_volume <= SQUID_VOL_THRESHOLD:
                    quantity = min(best_ask_volume, int((POSITION_LIMIT - current_position) * size_multiplier))
                    if quantity > 0:
                        orders_squid.append(Order(squid, best_ask, quantity))
            if squid_depth.buy_orders:
                best_bid = max(squid_depth.buy_orders.keys())
                best_bid_volume = squid_depth.buy_orders[best_bid]
                if best_bid >= squid_fair_value + effective_take_width_squid and best_bid_volume <= SQUID_VOL_THRESHOLD:
                    quantity = min(best_bid_volume, int((POSITION_LIMIT + current_position) * size_multiplier))
                    if quantity > 0:
                        orders_squid.append(Order(squid, best_bid, -quantity))
            # Market-making orders for squid ink
            if current_position < POSITION_LIMIT:
                buy_qty = int((POSITION_LIMIT - current_position) * size_multiplier)
                orders_squid.append(Order(squid, int(round(squid_fair_value - effective_mm_edge_squid)), buy_qty))
            if current_position > -POSITION_LIMIT:
                sell_qty = int((POSITION_LIMIT + current_position) * size_multiplier)
                orders_squid.append(Order(squid, int(round(squid_fair_value + effective_mm_edge_squid)), -sell_qty))
            # If we detect a strong downward trend (via EMA difference) and late in the day,
            # aggressively flatten long positions
            short_ema = trader_state.get("squid_short_ema", squid_fair_value)
            long_ema = trader_state.get("squid_long_ema", squid_fair_value)
            if state.timestamp >= LATE_DAY_TIMESTAMP and (short_ema - long_ema) < -SQUID_TREND_THRESHOLD:
                if current_position > 0:
                    orders_squid.append(Order(squid, int(round(squid_fair_value - 0.5 * effective_mm_edge_squid)), -current_position))
            result[squid] = orders_squid

        # Persist trader state for next iteration
        traderData = jsonpickle.encode(trader_state)
        conversions = 0
        return result, conversions, traderData
