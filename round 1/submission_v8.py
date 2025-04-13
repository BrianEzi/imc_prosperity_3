from datamodel import Order, TradingState, OrderDepth
from typing import List
import jsonpickle

class Trader:
    def __init__(self):

        HYPERPARAMS = {
            'KELP_BETA': 0.6811,
            'KELP_MM_EDGE': 1.0728,
            'KELP_TAKE_WIDTH': 2.4878,
            'LATE_DAY_SIZE_FACTOR': 0.7066,
            'LATE_DAY_SPREAD_FACTOR': 1.8936,
            'LATE_DAY_TIMESTAMP': 861778,
            'RESIN_MM_EDGE': 1.9429,
            'RESIN_TAKE_WIDTH': 8.2455,
            'SQUID_BETA': 0.5584,
            'SQUID_LONG_EMA_WINDOW': 36,
            'SQUID_MM_EDGE': 8.0081,
            'SQUID_SHORT_EMA_WINDOW': 6,
            'SQUID_TAKE_WIDTH': 6.4667,
            'SQUID_TREND_BIAS': 0.7824,
            'SQUID_TREND_THRESHOLD': 1.2344,
            'SQUID_VOL_THRESHOLD': 14,
        }
        
        # Define product identifiers
        self.product_resin = "RAINFOREST_RESIN"
        self.product_kelp = "KELP"
        self.product_squid = "SQUID_INK"

        # General parameters
        self.POSITION_LIMIT = 50

        # Late-day management (applied to all assets)
        self.LATE_DAY_TIMESTAMP = HYPERPARAMS['LATE_DAY_TIMESTAMP']
        self.LATE_DAY_SIZE_FACTOR = HYPERPARAMS['LATE_DAY_SIZE_FACTOR']
        self.LATE_DAY_SPREAD_FACTOR = HYPERPARAMS['LATE_DAY_SPREAD_FACTOR']

        # ------------------
        # Rainforest Resin (fixed fair value)
        # ------------------
        self.RESIN_FAIR_VALUE = 10000
        self.RESIN_TAKE_WIDTH = HYPERPARAMS['RESIN_TAKE_WIDTH']
        self.RESIN_MM_EDGE = HYPERPARAMS['RESIN_MM_EDGE']

        # ------------------
        # Kelp (dynamic fair value using EMA)
        # ------------------
        self.KELP_TAKE_WIDTH = HYPERPARAMS['KELP_TAKE_WIDTH']
        self.KELP_MM_EDGE = HYPERPARAMS['KELP_MM_EDGE']
        self.KELP_BETA = HYPERPARAMS['KELP_BETA']

        # ------------------
        # Squid Ink (momentum trading strategy)
        # ------------------
        self.SQUID_MOMENTUM_THRESHOLD = 10  # Basic threshold (tunable)
        # We still import some SQUID hyperparameters to set order levels
        self.SQUID_TAKE_WIDTH = HYPERPARAMS['SQUID_TAKE_WIDTH']
        self.SQUID_MM_EDGE = HYPERPARAMS['SQUID_MM_EDGE']
        self.SQUID_BETA = HYPERPARAMS['SQUID_BETA']
        self.SQUID_TREND_THRESHOLD = HYPERPARAMS['SQUID_TREND_THRESHOLD']
        self.SQUID_TREND_BIAS = HYPERPARAMS['SQUID_TREND_BIAS']
        self.SQUID_VOL_THRESHOLD = HYPERPARAMS['SQUID_VOL_THRESHOLD']
        self.SQUID_SHORT_EMA_WINDOW = HYPERPARAMS['SQUID_SHORT_EMA_WINDOW']
        self.SQUID_LONG_EMA_WINDOW = HYPERPARAMS['SQUID_LONG_EMA_WINDOW']
        # For momentum, we need to store the last midprice:
        self.initial_squid_momentum = None

    # ---------- Utility: EMA Update Function ----------
    def update_ema(self, last_ema, price, window):
        alpha = 2 / (window + 1)
        if last_ema is None:
            return price
        return alpha * price + (1 - alpha) * last_ema

    # ---------- Kelp: Dynamic Fair Value via EMA ----------
    def compute_kelp_fair_value(self, depth: OrderDepth, trader_state: dict) -> float:
        if not depth.buy_orders or not depth.sell_orders:
            return trader_state.get("kelp_last_price", 10000)
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        observed_mid = (best_bid + best_ask) / 2
        last_price = trader_state.get("kelp_last_price", observed_mid)
        new_fair = last_price + self.KELP_BETA * (observed_mid - last_price)
        trader_state["kelp_last_price"] = new_fair
        return new_fair

    # ---------- Late-day risk adjustment ----------
    def get_late_day_factor(self, timestamp: int) -> float:
        return self.LATE_DAY_SIZE_FACTOR if timestamp >= self.LATE_DAY_TIMESTAMP else 1.0

    # ---------- Basic Momentum for Squid Ink ----------
    def compute_squid_momentum(self, depth: OrderDepth, trader_state: dict) -> float:
        """
        Compute momentum for Squid Ink as the difference between the current midprice
        and the previous midprice stored in trader state.
        """
        if not depth.buy_orders or not depth.sell_orders:
            return 0
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        current_mid = (best_bid + best_ask) / 2
        last_mid = trader_state.get("squid_last_momentum", current_mid)
        momentum = current_mid - last_mid
        trader_state["squid_last_momentum"] = current_mid
        return momentum

    def run(self, state: TradingState):
        # Load persistent trader state
        trader_state = {}
        if state.traderData and state.traderData != "":
            trader_state = jsonpickle.decode(state.traderData)

        result = {}
        late_day_factor = self.get_late_day_factor(state.timestamp)

        # --------------------------
        # Process Rainforest Resin (Fixed Fair Value)
        # --------------------------
        resin = self.product_resin
        if resin in state.order_depths:
            resin_depth: OrderDepth = state.order_depths[resin]
            orders_resin: List[Order] = []
            current_position = state.position.get(resin, 0)
            fv = self.RESIN_FAIR_VALUE
            effective_take_width = self.RESIN_TAKE_WIDTH * (self.LATE_DAY_SPREAD_FACTOR if late_day_factor != 1.0 else 1)
            effective_mm_edge = self.RESIN_MM_EDGE * (self.LATE_DAY_SPREAD_FACTOR if late_day_factor != 1.0 else 1)
            size_multiplier = late_day_factor

            if resin_depth.sell_orders:
                best_ask = min(resin_depth.sell_orders.keys())
                vol = -resin_depth.sell_orders[best_ask]
                if best_ask <= fv - effective_take_width:
                    qty = min(vol, int((self.POSITION_LIMIT - current_position) * size_multiplier))
                    if qty > 0:
                        orders_resin.append(Order(resin, round(best_ask), qty))
            if resin_depth.buy_orders:
                best_bid = max(resin_depth.buy_orders.keys())
                vol = resin_depth.buy_orders[best_bid]
                if best_bid >= fv + effective_take_width:
                    qty = min(vol, int((self.POSITION_LIMIT + current_position) * size_multiplier))
                    if qty > 0:
                        orders_resin.append(Order(resin, round(best_bid), -qty))
            if current_position < self.POSITION_LIMIT:
                qty = int((self.POSITION_LIMIT - current_position) * size_multiplier)
                orders_resin.append(Order(resin, round(fv - effective_mm_edge), qty))
            if current_position > -self.POSITION_LIMIT:
                qty = int((self.POSITION_LIMIT + current_position) * size_multiplier)
                orders_resin.append(Order(resin, round(fv + effective_mm_edge), -qty))
            result[resin] = orders_resin

        # --------------------------
        # Process Kelp (Dynamic Fair Value)
        # --------------------------
        kelp = self.product_kelp
        if kelp in state.order_depths:
            kelp_depth: OrderDepth = state.order_depths[kelp]
            orders_kelp: List[Order] = []
            current_position = state.position.get(kelp, 0)
            fv = self.compute_kelp_fair_value(kelp_depth, trader_state)
            effective_take_width = self.KELP_TAKE_WIDTH * (self.LATE_DAY_SPREAD_FACTOR if late_day_factor != 1.0 else 1)
            effective_mm_edge = self.KELP_MM_EDGE * (self.LATE_DAY_SPREAD_FACTOR if late_day_factor != 1.0 else 1)
            size_multiplier = late_day_factor

            if kelp_depth.sell_orders:
                best_ask = min(kelp_depth.sell_orders.keys())
                vol = -kelp_depth.sell_orders[best_ask]
                if best_ask <= fv - effective_take_width:
                    qty = min(vol, int((self.POSITION_LIMIT - current_position) * size_multiplier))
                    if qty > 0:
                        orders_kelp.append(Order(kelp, best_ask, qty))
            if kelp_depth.buy_orders:
                best_bid = max(kelp_depth.buy_orders.keys())
                vol = kelp_depth.buy_orders[best_bid]
                if best_bid >= fv + effective_take_width:
                    qty = min(vol, int((self.POSITION_LIMIT + current_position) * size_multiplier))
                    if qty > 0:
                        orders_kelp.append(Order(kelp, best_bid, -qty))
            if current_position < self.POSITION_LIMIT:
                qty = int((self.POSITION_LIMIT - current_position) * size_multiplier)
                orders_kelp.append(Order(kelp, int(round(fv - effective_mm_edge)), qty))
            if current_position > -self.POSITION_LIMIT:
                qty = int((self.POSITION_LIMIT + current_position) * size_multiplier)
                orders_kelp.append(Order(kelp, int(round(fv + effective_mm_edge)), -qty))
            result[kelp] = orders_kelp

        # --------------------------
        # Process Squid Ink with Momentum Trading Strategy
        # --------------------------
        squid = self.product_squid
        if squid in state.order_depths:
            squid_depth: OrderDepth = state.order_depths[squid]
            orders_squid: List[Order] = []
            current_position = state.position.get(squid, 0)
            # Compute the current midprice
            if squid_depth.buy_orders and squid_depth.sell_orders:
                best_bid = max(squid_depth.buy_orders.keys())
                best_ask = min(squid_depth.sell_orders.keys())
                current_mid = (best_bid + best_ask) / 2
            else:
                current_mid = 5000  # Fallback initial value

            # Compute momentum as the change in midprice from last tick
            momentum = self.compute_squid_momentum(squid_depth, trader_state)
            
            # For a basic momentum strategy:
            #   - If momentum is positive (price rising strongly), we assume the upward trend will continue, so BUY.
            #   - If momentum is negative (price falling strongly), we assume the downward trend will continue, so SELL.
            # The order is placed at the current best price from the order book.
            if momentum > self.SQUID_MOMENTUM_THRESHOLD:
                # Upward momentum: buy using the best ask
                if squid_depth.sell_orders:
                    best_ask = min(squid_depth.sell_orders.keys())
                    ask_vol = -squid_depth.sell_orders[best_ask]
                    qty = min(ask_vol, self.POSITION_LIMIT - current_position)
                    if qty > 0:
                        orders_squid.append(Order(squid, best_ask, qty))
            elif momentum < -self.SQUID_MOMENTUM_THRESHOLD:
                # Downward momentum: sell using the best bid
                if squid_depth.buy_orders:
                    best_bid = max(squid_depth.buy_orders.keys())
                    bid_vol = squid_depth.buy_orders[best_bid]
                    qty = min(bid_vol, self.POSITION_LIMIT + current_position)
                    if qty > 0:
                        orders_squid.append(Order(squid, best_bid, -qty))
            # Additionally, add a basic market-making order near the current midprice.
            if current_position < self.POSITION_LIMIT:
                buy_qty = self.POSITION_LIMIT - current_position
                orders_squid.append(Order(squid, int(round(current_mid - 1)), buy_qty))
            if current_position > -self.POSITION_LIMIT:
                sell_qty = self.POSITION_LIMIT + current_position
                orders_squid.append(Order(squid, int(round(current_mid + 1)), -sell_qty))
            result[squid] = orders_squid

        traderData = jsonpickle.encode(trader_state)
        conversions = 0
        return result, conversions, traderData
