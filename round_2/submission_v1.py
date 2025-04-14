from datamodel import Order, TradingState, OrderDepth
from typing import List, Optional, Tuple
import jsonpickle

# Position limits for new assets
POSITION_LIMIT_CROISSANTS = 250
POSITION_LIMIT_JAMS = 350
POSITION_LIMIT_DJEMBES = 60
POSITION_LIMIT_BASKET1 = 60
POSITION_LIMIT_BASKET2 = 100

# A relative arbitrage threshold (e.g., 2% deviation)
ARBITRAGE_THRESHOLD = 0.02

class Trader:
    def __init__(self):
        # New asset names
        self.product_croissants = "CROISSANTS"
        self.product_jams = "JAMS"
        self.product_djembes = "DJEMBES"
        self.product_basket1 = "PICNIC_BASKET1"
        self.product_basket2 = "PICNIC_BASKET2"
        # We assume state.position uses these names

    def midprice(self, depth: OrderDepth) -> Optional[float]:
        """Return the midprice computed as the average of the best bid and best ask; if unavailable return None."""
        if not depth.buy_orders or not depth.sell_orders:
            return None
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        return (best_bid + best_ask) / 2

    def compute_component_fv(self, product: str, state: TradingState) -> Optional[float]:
        """Compute the fair value of a component using its midprice."""
        if product not in state.order_depths:
            return None
        return self.midprice(state.order_depths[product])

    def compute_basket_theoretical_value(self, state: TradingState) -> Tuple[Optional[float], Optional[float]]:
        """
        Compute the theoretical basket values:
         - PICNIC_BASKET1 = 6 * CROISSANTS + 3 * JAMS + 1 * DJEMBES
         - PICNIC_BASKET2 = 4 * CROISSANTS + 2 * JAMS
        """
        croissant_fv = self.compute_component_fv(self.product_croissants, state)
        jam_fv = self.compute_component_fv(self.product_jams, state)
        djembe_fv = self.compute_component_fv(self.product_djembes, state)
        
        if croissant_fv is None or jam_fv is None or djembe_fv is None:
            basket1_theory = None
        else:
            basket1_theory = 6 * croissant_fv + 3 * jam_fv + 1 * djembe_fv

        if croissant_fv is None or jam_fv is None:
            basket2_theory = None
        else:
            basket2_theory = 4 * croissant_fv + 2 * jam_fv

        return basket1_theory, basket2_theory

    def create_market_making_orders(self, product: str, fv: float, current_position: int, position_limit: int, spread: float) -> List[Order]:
        """
        Fallback market-making orders near the computed fair value.
        """
        orders = []
        if current_position < position_limit:
            qty = position_limit - current_position
            orders.append(Order(product, int(round(fv - spread)), qty))
        if current_position > -position_limit:
            qty = position_limit + current_position
            orders.append(Order(product, int(round(fv + spread)), -qty))
        return orders

    def available_volume(self, depth: OrderDepth, side: str) -> int:
        """
        Return the available volume on a given side ('buy' for bid, 'sell' for ask)
        at the best price level.
        """
        if side == "buy" and depth.buy_orders:
            best_bid = max(depth.buy_orders.keys())
            return depth.buy_orders[best_bid]
        elif side == "sell" and depth.sell_orders:
            best_ask = min(depth.sell_orders.keys())
            return -depth.sell_orders[best_ask]
        else:
            return 0

    def run(self, state: TradingState):
        # No need to carry a persistent trader state here for our basket arbitrage strategy.
        result = {}
        
        # --- Compute market midprices for components ---
        croissant_fv = self.compute_component_fv(self.product_croissants, state)
        jam_fv = self.compute_component_fv(self.product_jams, state)
        djembe_fv = self.compute_component_fv(self.product_djembes, state)
        
        # --- Compute market midprices for the baskets ---
        basket1_market = None
        basket2_market = None
        if self.product_basket1 in state.order_depths:
            basket1_market = self.midprice(state.order_depths[self.product_basket1])
        if self.product_basket2 in state.order_depths:
            basket2_market = self.midprice(state.order_depths[self.product_basket2])
        
        # Compute theoretical basket values from components
        basket1_theory, basket2_theory = self.compute_basket_theoretical_value(state)
        
        # For debugging: one might log these computed values.
        # print(f"Basket1 Market: {basket1_market}, Theory: {basket1_theory}")
        # print(f"Basket2 Market: {basket2_market}, Theory: {basket2_theory}")
        
        # --- Arbitrage Logic for PICNIC_BASKET1 ---
        orders_basket1 = []
        if basket1_market is not None and basket1_theory is not None:
            # Compute percentage deviation. Positive deviation means basket is underpriced.
            deviation = (basket1_theory - basket1_market) / basket1_theory

            # If basket is underpriced (deviation > threshold) then we want to buy basket and sell its components.
            if deviation > ARBITRAGE_THRESHOLD:
                depth_b1 = state.order_depths[self.product_basket1]
                best_ask = min(depth_b1.sell_orders.keys()) if depth_b1.sell_orders else basket1_market
                pos_b1 = state.position.get(self.product_basket1, 0)
                # Check if available volume on the basket side is sufficient.
                avail_b1 = self.available_volume(depth_b1, "sell")
                if avail_b1 >= 5:
                    qty = min(10, (POSITION_LIMIT_BASKET1 - pos_b1))
                    if qty > 0:
                        orders_basket1.append(Order(self.product_basket1, best_ask, qty))
                # Simultaneously, sell the corresponding components:
                # Sell 6 CROISSANTS
                if self.product_croissants in state.order_depths and croissant_fv is not None:
                    depth_c = state.order_depths[self.product_croissants]
                    best_bid_c = max(depth_c.buy_orders.keys()) if depth_c.buy_orders else croissant_fv
                    pos_c = state.position.get(self.product_croissants, 0)
                    avail_c = self.available_volume(depth_c, "buy")
                    if avail_c >= 10:
                        sell_qty = min(6 * qty, (POSITION_LIMIT_CROISSANTS + pos_c))
                        if sell_qty > 0:
                            orders_basket1.append(Order(self.product_croissants, best_bid_c, -sell_qty))
                # Sell 3 JAMS
                if self.product_jams in state.order_depths and jam_fv is not None:
                    depth_j = state.order_depths[self.product_jams]
                    best_bid_j = max(depth_j.buy_orders.keys()) if depth_j.buy_orders else jam_fv
                    pos_j = state.position.get(self.product_jams, 0)
                    avail_j = self.available_volume(depth_j, "buy")
                    if avail_j >= 10:
                        sell_qty = min(3 * qty, (POSITION_LIMIT_JAMS + pos_j))
                        if sell_qty > 0:
                            orders_basket1.append(Order(self.product_jams, best_bid_j, -sell_qty))
                # Sell 1 DJEMBE
                if self.product_djembes in state.order_depths and djembe_fv is not None:
                    depth_d = state.order_depths[self.product_djembes]
                    best_bid_d = max(depth_d.buy_orders.keys()) if depth_d.buy_orders else djembe_fv
                    pos_d = state.position.get(self.product_djembes, 0)
                    avail_d = self.available_volume(depth_d, "buy")
                    if avail_d >= 5:
                        sell_qty = min(1 * qty, (POSITION_LIMIT_DJEMBES + pos_d))
                        if sell_qty > 0:
                            orders_basket1.append(Order(self.product_djembes, best_bid_d, -sell_qty))
            # If basket is overpriced (deviation < -threshold): sell basket and buy components.
            elif basket1_market > basket1_theory * (1 + ARBITRAGE_THRESHOLD):
                depth_b1 = state.order_depths[self.product_basket1]
                best_bid = max(depth_b1.buy_orders.keys()) if depth_b1.buy_orders else basket1_market
                pos_b1 = state.position.get(self.product_basket1, 0)
                if self.available_volume(depth_b1, "buy") >= 5:
                    qty = min(10, (POSITION_LIMIT_BASKET1 + pos_b1))
                    if qty > 0:
                        orders_basket1.append(Order(self.product_basket1, best_bid, -qty))
                # Buy components:
                if self.product_croissants in state.order_depths and croissant_fv is not None:
                    depth_c = state.order_depths[self.product_croissants]
                    best_ask_c = min(depth_c.sell_orders.keys()) if depth_c.sell_orders else croissant_fv
                    pos_c = state.position.get(self.product_croissants, 0)
                    if self.available_volume(depth_c, "sell") >= 10:
                        buy_qty = min(6 * qty, (POSITION_LIMIT_CROISSANTS - pos_c))
                        if buy_qty > 0:
                            orders_basket1.append(Order(self.product_croissants, best_ask_c, buy_qty))
                if self.product_jams in state.order_depths and jam_fv is not None:
                    depth_j = state.order_depths[self.product_jams]
                    best_ask_j = min(depth_j.sell_orders.keys()) if depth_j.sell_orders else jam_fv
                    pos_j = state.position.get(self.product_jams, 0)
                    if self.available_volume(depth_j, "sell") >= 10:
                        buy_qty = min(3 * qty, (POSITION_LIMIT_JAMS - pos_j))
                        if buy_qty > 0:
                            orders_basket1.append(Order(self.product_jams, best_ask_j, buy_qty))
                if self.product_djembes in state.order_depths and djembe_fv is not None:
                    depth_d = state.order_depths[self.product_djembes]
                    best_ask_d = min(depth_d.sell_orders.keys()) if depth_d.sell_orders else djembe_fv
                    pos_d = state.position.get(self.product_djembes, 0)
                    if self.available_volume(depth_d, "sell") >= 5:
                        buy_qty = min(1 * qty, (POSITION_LIMIT_DJEMBES - pos_d))
                        if buy_qty > 0:
                            orders_basket1.append(Order(self.product_djembes, best_ask_d, buy_qty))
            # Fallback market-making orders for Basket1:
            pos_b1 = state.position.get(self.product_basket1, 0)
            mm_orders_b1 = []
            if basket1_market is not None:
                if pos_b1 < POSITION_LIMIT_BASKET1:
                    mm_orders_b1.append(Order(self.product_basket1, int(round(basket1_market - 1)), POSITION_LIMIT_BASKET1 - pos_b1))
                if pos_b1 > -POSITION_LIMIT_BASKET1:
                    mm_orders_b1.append(Order(self.product_basket1, int(round(basket1_market + 1)), -(POSITION_LIMIT_BASKET1 + pos_b1)))
            orders_basket1.extend(mm_orders_b1)
            result[self.product_basket1] = orders_basket1

        # --------------------------
        # Process Basket2 arbitrage (similar logic)
        # --------------------------
        orders_basket2 = []
        if self.product_basket2 in state.order_depths:
            basket2_market = self.midprice(state.order_depths[self.product_basket2])
            basket2_theory = None
            croissant_fv = self.compute_component_fv(self.product_croissants, state)
            jam_fv = self.compute_component_fv(self.product_jams, state)
            if croissant_fv is not None and jam_fv is not None:
                basket2_theory = 4 * croissant_fv + 2 * jam_fv
            if basket2_market is not None and basket2_theory is not None:
                if basket2_market < basket2_theory * (1 - ARBITRAGE_THRESHOLD):
                    depth_b2 = state.order_depths[self.product_basket2]
                    best_ask_b2 = min(depth_b2.sell_orders.keys()) if depth_b2.sell_orders else basket2_market
                    pos_b2 = state.position.get(self.product_basket2, 0)
                    if self.available_volume(depth_b2, "sell") >= 5:
                        qty = min(10, (POSITION_LIMIT_BASKET2 - pos_b2))
                        if qty > 0:
                            orders_basket2.append(Order(self.product_basket2, best_ask_b2, qty))
                    # Sell components (croissants and jams)
                    if self.product_croissants in state.order_depths and croissant_fv is not None:
                        depth_c = state.order_depths[self.product_croissants]
                        best_bid_c = max(depth_c.buy_orders.keys()) if depth_c.buy_orders else croissant_fv
                        pos_c = state.position.get(self.product_croissants, 0)
                        sell_qty = min(4 * qty, (POSITION_LIMIT_CROISSANTS + pos_c))
                        if sell_qty > 0:
                            orders_basket2.append(Order(self.product_croissants, best_bid_c, -sell_qty))
                    if self.product_jams in state.order_depths and jam_fv is not None:
                        depth_j = state.order_depths[self.product_jams]
                        best_bid_j = max(depth_j.buy_orders.keys()) if depth_j.buy_orders else jam_fv
                        pos_j = state.position.get(self.product_jams, 0)
                        sell_qty = min(2 * qty, (POSITION_LIMIT_JAMS + pos_j))
                        if sell_qty > 0:
                            orders_basket2.append(Order(self.product_jams, best_bid_j, -sell_qty))
                elif basket2_market > basket2_theory * (1 + ARBITRAGE_THRESHOLD):
                    depth_b2 = state.order_depths[self.product_basket2]
                    best_bid_b2 = max(depth_b2.buy_orders.keys()) if depth_b2.buy_orders else basket2_market
                    pos_b2 = state.position.get(self.product_basket2, 0)
                    if self.available_volume(depth_b2, "buy") >= 5:
                        qty = min(10, (POSITION_LIMIT_BASKET2 + pos_b2))
                        if qty > 0:
                            orders_basket2.append(Order(self.product_basket2, best_bid_b2, -qty))
                    # Buy components:
                    if self.product_croissants in state.order_depths and croissant_fv is not None:
                        depth_c = state.order_depths[self.product_croissants]
                        best_ask_c = min(depth_c.sell_orders.keys()) if depth_c.sell_orders else croissant_fv
                        pos_c = state.position.get(self.product_croissants, 0)
                        buy_qty = min(4 * qty, (POSITION_LIMIT_CROISSANTS - pos_c))
                        if buy_qty > 0:
                            orders_basket2.append(Order(self.product_croissants, best_ask_c, buy_qty))
                    if self.product_jams in state.order_depths and jam_fv is not None:
                        depth_j = state.order_depths[self.product_jams]
                        best_ask_j = min(depth_j.sell_orders.keys()) if depth_j.sell_orders else jam_fv
                        pos_j = state.position.get(self.product_jams, 0)
                        buy_qty = min(2 * qty, (POSITION_LIMIT_JAMS - pos_j))
                        if buy_qty > 0:
                            orders_basket2.append(Order(self.product_jams, best_ask_j, buy_qty))
            # Fallback market-making orders for Basket2
            pos_b2 = state.position.get(self.product_basket2, 0)
            mm_orders_b2 = []
            if basket2_market is not None:
                if pos_b2 < POSITION_LIMIT_BASKET2:
                    mm_orders_b2.append(Order(self.product_basket2, int(round(basket2_market - 1)), POSITION_LIMIT_BASKET2 - pos_b2))
                if pos_b2 > -POSITION_LIMIT_BASKET2:
                    mm_orders_b2.append(Order(self.product_basket2, int(round(basket2_market + 1)), -(POSITION_LIMIT_BASKET2 + pos_b2)))
            orders_basket2.extend(mm_orders_b2)
            result[self.product_basket2] = orders_basket2

        # --------------------------
        # Process individual component assets (Croissants, Jams, Djembes)
        # Also issue basic market-making orders so that we remain active.
        # --------------------------
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

        traderData = jsonpickle.encode({})
        return result, 0, traderData
