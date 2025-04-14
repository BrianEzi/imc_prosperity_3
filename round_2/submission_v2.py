from datamodel import Order, TradingState, OrderDepth
from typing import List, Optional, Tuple
import jsonpickle
import numpy as np

# --- Constants: (These could be placed in a config file.) ---
POSITION_LIMIT_BASKET1 = 60
POSITION_LIMIT_CROISSANTS = 250
POSITION_LIMIT_JAMS = 350
POSITION_LIMIT_DJEMBES = 60
ARBITRAGE_QUANTITY = 10
DEFAULT_SPREAD_MEAN = 70.0
ROLLING_WINDOW = 30
ZSCORE_THRESHOLD = 7.0

class Trader:
    def __init__(self):
        # New asset names
        self.product_croissants = "CROISSANTS"
        self.product_jams = "JAMS"
        self.product_djembes = "DJEMBES"
        self.product_basket1 = "PICNIC_BASKET1"
        
        # You can add further initialization as needed.

    def midprice(self, depth: OrderDepth) -> Optional[float]:
        """Return midprice as average of best bid and best ask, if available; else None."""
        if not depth.buy_orders or not depth.sell_orders:
            return None
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        return (best_bid + best_ask) / 2

    def compute_component_fv(self, product: str, state: TradingState) -> Optional[float]:
        """Get the midprice (fair value) for an individual asset from its order depth."""
        if product not in state.order_depths:
            return None
        return self.midprice(state.order_depths[product])

    def compute_basket_theoretical_value(self, state: TradingState) -> Tuple[Optional[float], Optional[float]]:
        """
        Compute the theoretical values for the baskets:
          PICNIC_BASKET1 = 6 * CROISSANTS + 3 * JAMS + 1 * DJEMBES
          PICNIC_BASKET2 = 4 * CROISSANTS + 2 * JAMS    (unused here)
        Returns a tuple (basket1_theory, basket2_theory)
        """
        fv_c = self.compute_component_fv(self.product_croissants, state)
        fv_j = self.compute_component_fv(self.product_jams, state)
        fv_d = self.compute_component_fv(self.product_djembes, state)
        if fv_c is None or fv_j is None or fv_d is None:
            basket1_theory = None
        else:
            basket1_theory = 6 * fv_c + 3 * fv_j + fv_d

        # Not used here, but provided for completeness:
        if fv_c is None or fv_j is None:
            basket2_theory = None
        else:
            basket2_theory = 4 * fv_c + 2 * fv_j

        return basket1_theory, basket2_theory

    def update_spread_history(self, trader_state: dict, spread: float) -> List[float]:
        """Maintain and return a rolling history of spread values."""
        if "spread_history" not in trader_state:
            trader_state["spread_history"] = []
        trader_state["spread_history"].append(spread)
        if len(trader_state["spread_history"]) > ROLLING_WINDOW:
            trader_state["spread_history"].pop(0)
        return trader_state["spread_history"]

    def compute_modified_zscore(self, spread: float, history: List[float]) -> Optional[float]:
        """Compute modified z-score using a hardcoded mean and rolling standard deviation."""
        if len(history) < ROLLING_WINDOW:
            return None
        rolling_std = np.std(history)
        if rolling_std == 0:
            return 0.0
        return (spread - DEFAULT_SPREAD_MEAN) / rolling_std

    def create_market_making_orders(self, product: str, fv: float, current_position: int, position_limit: int, spread: float) -> List[Order]:
        """Generate simple market-making orders near the given fair value."""
        orders = []
        if current_position < position_limit:
            qty = position_limit - current_position
            orders.append(Order(product, int(round(fv - spread)), qty))
        if current_position > -position_limit:
            qty = position_limit + current_position
            orders.append(Order(product, int(round(fv + spread)), -qty))
        return orders

    def execute_arbitrage_for_basket1(self, state: TradingState, trader_state: dict) -> List[Order]:
        """Calculate the spread and generate arbitrage orders for PICNIC_BASKET1."""
        orders_arbitrage = []
        # Ensure we have an order depth for Basket1
        if self.product_basket1 not in state.order_depths:
            return orders_arbitrage
        
        depth_basket = state.order_depths[self.product_basket1]
        basket_mid = self.midprice(depth_basket)
        if basket_mid is None:
            return orders_arbitrage

        # Compute theoretical value for Basket1
        basket1_theory, _ = self.compute_basket_theoretical_value(state)
        if basket1_theory is None:
            return orders_arbitrage

        # Compute the spread and update rolling history
        spread = basket_mid - basket1_theory
        self.update_spread_history(trader_state, spread)
        if len(trader_state["spread_history"]) < ROLLING_WINDOW:
            # Not enough history—fallback to market making
            return self.create_market_making_orders(
                self.product_basket1, basket_mid, state.position.get(self.product_basket1, 0), POSITION_LIMIT_BASKET1, 1
            )
        rolling_std = np.std(trader_state["spread_history"])
        zscore = (spread - DEFAULT_SPREAD_MEAN) / rolling_std if rolling_std > 0 else 0.0

        # Debug: Print computed spread and z-score
        # print(f"Basket1 mid: {basket_mid}, Theory: {basket1_theory}, Spread: {spread}, Z-score: {zscore}")

        pos_basket = state.position.get(self.product_basket1, 0)
        if zscore > ZSCORE_THRESHOLD:
            # Spread is too high: Basket1 is expensive → Sell basket and buy components
            best_bid_b1 = max(depth_basket.buy_orders.keys()) if depth_basket.buy_orders else basket_mid
            qty = min(ARBITRAGE_QUANTITY, (POSITION_LIMIT_BASKET1 + pos_basket))
            if qty > 0:
                orders_arbitrage.append(Order(self.product_basket1, best_bid_b1, -qty))
            # Buy components: CROISSANTS, JAMS, DJEMBES at their best ask prices.
            fv_c = self.compute_component_fv(self.product_croissants, state)
            if self.product_croissants in state.order_depths and fv_c is not None:
                depth_c = state.order_depths[self.product_croissants]
                best_ask_c = min(depth_c.sell_orders.keys()) if depth_c.sell_orders else fv_c
                orders_arbitrage.append(Order(self.product_croissants, best_ask_c, 6 * qty))
            fv_j = self.compute_component_fv(self.product_jams, state)
            if self.product_jams in state.order_depths and fv_j is not None:
                depth_j = state.order_depths[self.product_jams]
                best_ask_j = min(depth_j.sell_orders.keys()) if depth_j.sell_orders else fv_j
                orders_arbitrage.append(Order(self.product_jams, best_ask_j, 3 * qty))
            fv_d = self.compute_component_fv(self.product_djembes, state)
            if self.product_djembes in state.order_depths and fv_d is not None:
                depth_d = state.order_depths[self.product_djembes]
                best_ask_d = min(depth_d.sell_orders.keys()) if depth_d.sell_orders else fv_d
                orders_arbitrage.append(Order(self.product_djembes, best_ask_d, 1 * qty))
        elif zscore < -ZSCORE_THRESHOLD:
            # Spread is too low: Basket1 is cheap → Buy basket and sell components
            best_ask_b1 = min(depth_basket.sell_orders.keys()) if depth_basket.sell_orders else basket_mid
            qty = min(ARBITRAGE_QUANTITY, (POSITION_LIMIT_BASKET1 - pos_basket))
            if qty > 0:
                orders_arbitrage.append(Order(self.product_basket1, best_ask_b1, qty))
            # Sell components at best bid prices.
            fv_c = self.compute_component_fv(self.product_croissants, state)
            if self.product_croissants in state.order_depths and fv_c is not None:
                depth_c = state.order_depths[self.product_croissants]
                best_bid_c = max(depth_c.buy_orders.keys()) if depth_c.buy_orders else fv_c
                orders_arbitrage.append(Order(self.product_croissants, best_bid_c, -6 * qty))
            fv_j = self.compute_component_fv(self.product_jams, state)
            if self.product_jams in state.order_depths and fv_j is not None:
                depth_j = state.order_depths[self.product_jams]
                best_bid_j = max(depth_j.buy_orders.keys()) if depth_j.buy_orders else fv_j
                orders_arbitrage.append(Order(self.product_jams, best_bid_j, -3 * qty))
            fv_d = self.compute_component_fv(self.product_djembes, state)
            if self.product_djembes in state.order_depths and fv_d is not None:
                depth_d = state.order_depths[self.product_djembes]
                best_bid_d = max(depth_d.buy_orders.keys()) if depth_d.buy_orders else fv_d
                orders_arbitrage.append(Order(self.product_djembes, best_bid_d, -1 * qty))
        else:
            # No strong arbitrage signal: fallback to market-making orders near basket midprice.
            orders_arbitrage = self.create_market_making_orders(
                self.product_basket1, basket_mid, state.position.get(self.product_basket1, 0), POSITION_LIMIT_BASKET1, 1
            )
        return orders_arbitrage

    def run(self, state: TradingState):
        # Initialize or decode trader state
        trader_state = {}
        if state.traderData and state.traderData != "":
            trader_state = jsonpickle.decode(state.traderData)
        result = {}

        # --- Process PICNIC_BASKET1 using spread-arbitrage ---
        if self.product_basket1 in state.order_depths:
            basket1_mid = self.midprice(state.order_depths[self.product_basket1])
        else:
            basket1_mid = None

        # Compute theoretical basket value for PICNIC_BASKET1
        basket1_theory = self.compute_basket_theoretical_value(state)[0]
        
        if basket1_mid is not None and basket1_theory is not None:
            arb_orders = self.execute_arbitrage_for_basket1(state, trader_state)
            result[self.product_basket1] = arb_orders
        else:
            result[self.product_basket1] = self.create_market_making_orders(
                self.product_basket1, basket1_mid, state.position.get(self.product_basket1, 0), POSITION_LIMIT_BASKET1, 1
            )

        # --- Process individual components (fallback market making) ---
        for prod, pos_limit in [(self.product_croissants, POSITION_LIMIT_CROISSANTS),
                                (self.product_jams, POSITION_LIMIT_JAMS),
                                (self.product_djembes, POSITION_LIMIT_DJEMBES)]:
            if prod in state.order_depths:
                orders = []
                depth = state.order_depths[prod]
                fv = self.midprice(depth)
                if fv is None:
                    continue
                pos = state.position.get(prod, 0)
                if pos < pos_limit:
                    qty = pos_limit - pos
                    orders.append(Order(prod, int(round(fv - 1)), qty))
                if pos > -pos_limit:
                    qty = pos_limit + pos
                    orders.append(Order(prod, int(round(fv + 1)), -qty))
                result[prod] = orders

        traderData = jsonpickle.encode(trader_state)
        return result, 0, traderData

