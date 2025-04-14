from datamodel import Order, TradingState, OrderDepth
from typing import List
import jsonpickle

class Trader:
    def __init__(self):
        # Define product identifiers
        self.product_resin = "RAINFOREST_RESIN"
        self.product_kelp = "KELP"

        # Fixed parameters for Rainforest Resin
        self.resin_fair_value = 10000      # Fixed fair value for Rainforest Resin
        self.resin_position_limit = 50     # Position limit for Resin
        self.resin_take_width = 5          # Take width for Resin
        self.resin_mm_edge = 2             # Market-making edge for Resin

        # Parameters for Kelp
        self.kelp_position_limit = 50      # Position limit for Kelp
        self.kelp_take_width = 5           # Take width for Kelp (tunable)
        self.kelp_mm_edge = 1.5              # Market making edge for Kelp (tunable)
        self.kelp_beta = 0.25               # Smoothing parameter for dynamic fair value adjustment

    def compute_kelp_fair_value(self, order_depth: OrderDepth, trader_state: dict) -> float:
        """
        Compute a dynamic fair value for Kelp based on the current order book.
        Uses an exponential moving average-like update based on the observed midprice.
        """
        # Ensure both sides of the order book exist
        if not order_depth.buy_orders or not order_depth.sell_orders:
            # If one side is empty, fallback to previous value if available
            return trader_state.get("kelp_last_price", 10000)  # Fallback initial value
        
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        observed_mid = (best_bid + best_ask) / 2

        # Retrieve last fair value for kelp, or initialize with observed mid
        last_price = trader_state.get("kelp_last_price", observed_mid)
        new_fair = last_price + self.kelp_beta * (observed_mid - last_price)
        trader_state["kelp_last_price"] = new_fair
        return new_fair

    def run(self, state: TradingState):
        # Load persistent trader state
        trader_state = {}
        if state.traderData and state.traderData != "":
            trader_state = jsonpickle.decode(state.traderData)

        result = {}

        # --------------------------
        # Process Rainforest Resin
        # --------------------------
        resin = self.product_resin
        if resin in state.order_depths:
            order_depth: OrderDepth = state.order_depths[resin]
            orders_resin: List[Order] = []
            current_position = state.position.get(resin, 0)

            # Take orders if price is favorable relative to the fixed fair value
            if order_depth.sell_orders:
                best_ask = min(order_depth.sell_orders.keys())
                best_ask_volume = -order_depth.sell_orders[best_ask]
                if best_ask <= self.resin_fair_value - self.resin_take_width:
                    quantity = min(best_ask_volume, self.resin_position_limit - current_position)
                    if quantity > 0:
                        orders_resin.append(Order(resin, best_ask, quantity))
            if order_depth.buy_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_bid_volume = order_depth.buy_orders[best_bid]
                if best_bid >= self.resin_fair_value + self.resin_take_width:
                    quantity = min(best_bid_volume, self.resin_position_limit + current_position)
                    if quantity > 0:
                        orders_resin.append(Order(resin, best_bid, -quantity))

            # Market-making orders for Resin
            if current_position < self.resin_position_limit:
                buy_qty = self.resin_position_limit - current_position
                orders_resin.append(Order(resin, self.resin_fair_value - self.resin_mm_edge, buy_qty))
            if current_position > -self.resin_position_limit:
                sell_qty = self.resin_position_limit + current_position
                orders_resin.append(Order(resin, self.resin_fair_value + self.resin_mm_edge, -sell_qty))

            result[resin] = orders_resin

        # --------------------------
        # Process Kelp with Dynamic Fair Value
        # --------------------------
        kelp = self.product_kelp
        if kelp in state.order_depths:
            order_depth: OrderDepth = state.order_depths[kelp]
            orders_kelp: List[Order] = []
            current_position = state.position.get(kelp, 0)

            kelp_fair_value = self.compute_kelp_fair_value(order_depth, trader_state)

            # Take orders for Kelp
            if order_depth.sell_orders:
                best_ask = min(order_depth.sell_orders.keys())
                best_ask_volume = -order_depth.sell_orders[best_ask]
                if best_ask <= kelp_fair_value - self.kelp_take_width:
                    quantity = min(best_ask_volume, self.kelp_position_limit - current_position)
                    if quantity > 0:
                        orders_kelp.append(Order(kelp, best_ask, quantity))
            if order_depth.buy_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_bid_volume = order_depth.buy_orders[best_bid]
                if best_bid >= kelp_fair_value + self.kelp_take_width:
                    quantity = min(best_bid_volume, self.kelp_position_limit + current_position)
                    if quantity > 0:
                        orders_kelp.append(Order(kelp, best_bid, -quantity))
            
            # Market-making orders for Kelp using dynamic fair value
            if current_position < self.kelp_position_limit:
                buy_qty = self.kelp_position_limit - current_position
                orders_kelp.append(Order(kelp, int(round(kelp_fair_value - self.kelp_mm_edge)), buy_qty))
            if current_position > -self.kelp_position_limit:
                sell_qty = self.kelp_position_limit + current_position
                orders_kelp.append(Order(kelp, int(round(kelp_fair_value + self.kelp_mm_edge)), -sell_qty))

            result[kelp] = orders_kelp

        # Persist updated trader state
        traderData = jsonpickle.encode(trader_state)
        conversions = 0  # No conversion logic in this example

        return result, conversions, traderData
