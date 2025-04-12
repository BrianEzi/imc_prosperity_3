from datamodel import Order, TradingState, OrderDepth
from typing import List
import jsonpickle

class Trader:
    def __init__(self):
        # Define product identifiers
        self.product_resin = "RAINFOREST_RESIN"
        self.product_kelp = "KELP"
        self.product_squid = "SQUID_INK"

        # Fixed parameters for Rainforest Resin
        self.resin_fair_value = 10000      # Fixed fair value for Rainforest Resin
        self.resin_position_limit = 50     # Position limit for Resin
        self.resin_take_width = 5          # Take width for Resin
        self.resin_mm_edge = 2             # Base market-making edge for Resin

        # Parameters for Kelp
        self.kelp_position_limit = 50      # Position limit for Kelp
        self.kelp_take_width = 5           # Base take width for Kelp
        self.kelp_mm_edge = 2              # Base market-making edge for Kelp
        self.kelp_beta = 0.62               # Smoothing parameter for dynamic fair value update

        # Parameters for Squid Ink
        self.squid_position_limit = 50     # Position limit for Squid Ink
        self.squid_take_width = 6.7          # Base take width for Squid Ink
        self.squid_mm_edge = 9.65             # Base market-making edge for Squid Ink
        self.squid_beta = 0.56              # Smoothing parameter for Squid Ink fair value update
        self.squid_vol_threshold = 7      # Volume threshold for immediate order taking
        self.squid_volatility_threshold = 50  # Threshold for high volatility regime (absolute deviation)

        # Risk adjustment factors when volatility is high
        self.volatility_spread_multiplier = 1.5  # Increase mm_edge by this factor when volatility is high
        self.volatility_size_multiplier = 0.75   # Reduce order sizes by this multiplier when volatility is high

    def compute_kelp_fair_value(self, order_depth: OrderDepth, trader_state: dict) -> float:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return trader_state.get("kelp_last_price", 10000)
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        observed_mid = (best_bid + best_ask) / 2
        last_price = trader_state.get("kelp_last_price", observed_mid)
        new_fair = last_price + self.kelp_beta * (observed_mid - last_price)
        trader_state["kelp_last_price"] = new_fair
        return new_fair

    def compute_squid_fair_value(self, order_depth: OrderDepth, trader_state: dict) -> float:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return trader_state.get("squid_last_price", 5000)
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        observed_mid = (best_bid + best_ask) / 2
        last_price = trader_state.get("squid_last_price", observed_mid)
        alpha = self.squid_beta
        new_fair = (1 - alpha) * last_price + alpha * observed_mid
        max_change = 1000
        change = new_fair - last_price
        if abs(change) > max_change:
            change = max_change if change > 0 else -max_change
            new_fair = last_price + change
        trader_state["squid_last_price"] = new_fair
        return new_fair

    def compute_squid_volatility(self, order_depth: OrderDepth, trader_state: dict) -> float:
        """
        Compute a simple volatility measure for Squid Ink as the absolute
        difference between the current observed midprice and the last stored midprice.
        """
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 0
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        current_mid = (best_bid + best_ask) / 2
        last_mid = trader_state.get("squid_last_mid", current_mid)
        volatility = abs(current_mid - last_mid)
        trader_state["squid_last_mid"] = current_mid
        return volatility

    def run(self, state: TradingState):
        # Load persistent trader state
        trader_state = {}
        if state.traderData and state.traderData != "":
            trader_state = jsonpickle.decode(state.traderData)
        
        result = {}

        # --------------------------
        # Process Squid Ink: Always trade Squid Ink with dynamic fair value,
        # but compute volatility to adjust aggressiveness.
        # --------------------------
        squid = self.product_squid
        squid_fair_value = None
        squid_volatility = 0
        if squid in state.order_depths:
            squid_depth: OrderDepth = state.order_depths[squid]
            squid_fair_value = self.compute_squid_fair_value(squid_depth, trader_state)
            squid_volatility = self.compute_squid_volatility(squid_depth, trader_state)
        # Determine volatility factor. If volatility exceeds threshold, scale up edge and reduce order sizes.
        if squid_volatility >= self.squid_volatility_threshold:
            spread_multiplier = self.volatility_spread_multiplier
            size_multiplier = self.volatility_size_multiplier
        else:
            spread_multiplier = 1
            size_multiplier = 1

        # Build Squid Ink orders (order taking and market-making)
        if squid in state.order_depths:
            orders_squid: List[Order] = []
            current_position = state.position.get(squid, 0)
            # Order taking for squid ink with volume filtering remains
            if squid_depth.sell_orders:
                best_ask = min(squid_depth.sell_orders.keys())
                best_ask_volume = -squid_depth.sell_orders[best_ask]
                if best_ask <= squid_fair_value - self.squid_take_width and best_ask_volume <= self.squid_vol_threshold:
                    quantity = min(best_ask_volume, self.squid_position_limit - current_position)
                    if quantity > 0:
                        orders_squid.append(Order(squid, best_ask, quantity))
            if squid_depth.buy_orders:
                best_bid = max(squid_depth.buy_orders.keys())
                best_bid_volume = squid_depth.buy_orders[best_bid]
                if best_bid >= squid_fair_value + self.squid_take_width and best_bid_volume <= self.squid_vol_threshold:
                    quantity = min(best_bid_volume, self.squid_position_limit + current_position)
                    if quantity > 0:
                        orders_squid.append(Order(squid, best_bid, -quantity))
            # Market-making orders for squid ink with adjusted edge if high volatility
            if current_position < self.squid_position_limit:
                buy_qty = int((self.squid_position_limit - current_position) * size_multiplier)
                orders_squid.append(Order(squid, int(round(squid_fair_value - self.squid_mm_edge * spread_multiplier)), buy_qty))
            if current_position > -self.squid_position_limit:
                sell_qty = int((self.squid_position_limit + current_position) * size_multiplier)
                orders_squid.append(Order(squid, int(round(squid_fair_value + self.squid_mm_edge * spread_multiplier)), -sell_qty))
            result[squid] = orders_squid

        # --------------------------
        # Process Rainforest Resin (fixed fair value strategy) with risk adjustment based on squid volatility
        # --------------------------
        resin = self.product_resin
        if resin in state.order_depths:
            resin_depth: OrderDepth = state.order_depths[resin]
            orders_resin: List[Order] = []
            current_position = state.position.get(resin, 0)
            # Use fixed fair value, but adjust effective mm_edge and take width if squid volatility is high.
            effective_resin_take_width = self.resin_take_width * spread_multiplier
            effective_resin_mm_edge = self.resin_mm_edge * spread_multiplier
            if resin_depth.sell_orders:
                best_ask = min(resin_depth.sell_orders.keys())
                best_ask_volume = -resin_depth.sell_orders[best_ask]
                if best_ask <= self.resin_fair_value - effective_resin_take_width:
                    quantity = min(best_ask_volume, int((self.resin_position_limit - current_position) * size_multiplier))
                    if quantity > 0:
                        orders_resin.append(Order(resin, best_ask, quantity))
            if resin_depth.buy_orders:
                best_bid = max(resin_depth.buy_orders.keys())
                best_bid_volume = resin_depth.buy_orders[best_bid]
                if best_bid >= self.resin_fair_value + effective_resin_take_width:
                    quantity = min(best_bid_volume, int((self.resin_position_limit + current_position) * size_multiplier))
                    if quantity > 0:
                        orders_resin.append(Order(resin, best_bid, -quantity))
            if current_position < self.resin_position_limit:
                buy_qty = int((self.resin_position_limit - current_position) * size_multiplier)
                orders_resin.append(Order(resin, self.resin_fair_value - effective_resin_mm_edge, buy_qty))
            if current_position > -self.resin_position_limit:
                sell_qty = int((self.resin_position_limit + current_position) * size_multiplier)
                orders_resin.append(Order(resin, self.resin_fair_value + effective_resin_mm_edge, -sell_qty))
            result[resin] = orders_resin

        # --------------------------
        # Process Kelp (dynamic fair value strategy) with risk adjustment based on squid volatility
        # --------------------------
        kelp = self.product_kelp
        if kelp in state.order_depths:
            kelp_depth: OrderDepth = state.order_depths[kelp]
            orders_kelp: List[Order] = []
            current_position = state.position.get(kelp, 0)
            kelp_fair_value = self.compute_kelp_fair_value(kelp_depth, trader_state)
            effective_kelp_take_width = self.kelp_take_width * spread_multiplier
            effective_kelp_mm_edge = self.kelp_mm_edge * spread_multiplier
            if kelp_depth.sell_orders:
                best_ask = min(kelp_depth.sell_orders.keys())
                best_ask_volume = -kelp_depth.sell_orders[best_ask]
                if best_ask <= kelp_fair_value - effective_kelp_take_width:
                    quantity = min(best_ask_volume, int((self.kelp_position_limit - current_position) * size_multiplier))
                    if quantity > 0:
                        orders_kelp.append(Order(kelp, best_ask, quantity))
            if kelp_depth.buy_orders:
                best_bid = max(kelp_depth.buy_orders.keys())
                best_bid_volume = kelp_depth.buy_orders[best_bid]
                if best_bid >= kelp_fair_value + effective_kelp_take_width:
                    quantity = min(best_bid_volume, int((self.kelp_position_limit + current_position) * size_multiplier))
                    if quantity > 0:
                        orders_kelp.append(Order(kelp, best_bid, -quantity))
            if current_position < self.kelp_position_limit:
                buy_qty = int((self.kelp_position_limit - current_position) * size_multiplier)
                orders_kelp.append(Order(kelp, int(round(kelp_fair_value - effective_kelp_mm_edge)), buy_qty))
            if current_position > -self.kelp_position_limit:
                sell_qty = int((self.kelp_position_limit + current_position) * size_multiplier)
                orders_kelp.append(Order(kelp, int(round(kelp_fair_value + effective_kelp_mm_edge)), -sell_qty))
            result[kelp] = orders_kelp

        # Persist updated trader state
        traderData = jsonpickle.encode(trader_state)
        conversions = 0
        return result, conversions, traderData
