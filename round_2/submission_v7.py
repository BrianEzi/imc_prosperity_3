import jsonpickle
from datamodel import Order, TradingState, OrderDepth
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

# --- Constants (defaults or overridden by hyperparameters) ---
POSITION_LIMIT_BASKET1 = 60
POSITION_LIMIT_BASKET2 = 120
POSITION_LIMIT_CROISSANTS = 250
POSITION_LIMIT_JAMS = 350
POSITION_LIMIT_DJEMBES = 60
MAX_SAFE_ORDER_SIZE = 1
MIN_ORDER_VOLUME = 0 # minimum volume required for order execution

try:
    from best_hyperparams_v4 import BEST_HYPERPARAMS as CONFIG_HYPERPARAMS
except ImportError:
    CONFIG_HYPERPARAMS = {}

#Use hyperparameters from CONFIG_HYPERPARAMS if available:
ARBITRAGE_QUANTITY = CONFIG_HYPERPARAMS.get("ARBITRAGE_QUANTITY", 10)
ROLLING_WINDOW = CONFIG_HYPERPARAMS.get("ROLLING_WINDOW", 30)
ZSCORE_THRESHOLD = CONFIG_HYPERPARAMS.get("ZSCORE_THRESHOLD", 7.0)

class Trader:
    def __init__(self):
        """Initialize parameters and hyperparameters for strategies."""
        # --- Basket & Arbitrage Product Names ---
        self.product_croissants = "CROISSANTS"
        self.product_jams = "JAMS"
        self.product_djembes = "DJEMBES"
        self.product_basket1 = "PICNIC_BASKET1"
        self.product_basket2 = "PICNIC_BASKET2"

        # --- Other Assets ---
        self.product_resin = "RAINFOREST_RESIN"
        self.product_kelp = "KELP"
        self.product_squid = "SQUID_INK"

        # --- Position Limits ---
        self.POS_LIMIT_BASKET1 = POSITION_LIMIT_BASKET1
        self.POS_LIMIT_BASKET2 = POSITION_LIMIT_BASKET2
        self.POS_LIMIT_CROISSANTS = POSITION_LIMIT_CROISSANTS
        self.POS_LIMIT_JAMS = POSITION_LIMIT_JAMS
        self.POS_LIMIT_DJEMBES = POSITION_LIMIT_DJEMBES
        # For individual assets (resin, kelp, squid)
        self.POSITION_LIMIT = 50

        # --- Hyperparameters for Basket Arbitrage ---
        self.rolling_window = CONFIG_HYPERPARAMS.get("ROLLING_WINDOW", ROLLING_WINDOW)
        self.zscore_threshold = CONFIG_HYPERPARAMS.get("ZSCORE_THRESHOLD", ZSCORE_THRESHOLD)

        # --- Hyperparameters for Resin, Kelp, and Squid ---
        # Resin parameters (fixed fair value strategy)
        self.RESIN_FAIR_VALUE = 10000
        self.RESIN_TAKE_WIDTH = CONFIG_HYPERPARAMS.get("RESIN_TAKE_WIDTH", 8.2455)
        self.RESIN_MM_EDGE = CONFIG_HYPERPARAMS.get("RESIN_MM_EDGE", 1.9429)

        # Kelp parameters (dynamic fair value strategy)
        self.KELP_TAKE_WIDTH = CONFIG_HYPERPARAMS.get("KELP_TAKE_WIDTH", 2.4878)
        self.KELP_MM_EDGE = CONFIG_HYPERPARAMS.get("KELP_MM_EDGE", 1.0728)
        self.KELP_BETA = CONFIG_HYPERPARAMS.get("KELP_BETA", 0.6811)

        # Squid parameters (momentum strategy)
        self.SQUID_TAKE_WIDTH = CONFIG_HYPERPARAMS.get("SQUID_TAKE_WIDTH", 6.4667)
        self.SQUID_MM_EDGE = CONFIG_HYPERPARAMS.get("SQUID_MM_EDGE", 8.0081)
        self.SQUID_BETA = CONFIG_HYPERPARAMS.get("SQUID_BETA", 0.5584)
        self.SQUID_TREND_THRESHOLD = CONFIG_HYPERPARAMS.get("SQUID_TREND_THRESHOLD", 1.2344)
        self.SQUID_TREND_BIAS = CONFIG_HYPERPARAMS.get("SQUID_TREND_BIAS", 0.7824)
        self.SQUID_VOL_THRESHOLD = CONFIG_HYPERPARAMS.get("SQUID_VOL_THRESHOLD", 14)
        self.SQUID_SHORT_EMA_WINDOW = CONFIG_HYPERPARAMS.get("SQUID_SHORT_EMA_WINDOW", 6)
        self.SQUID_LONG_EMA_WINDOW = CONFIG_HYPERPARAMS.get("SQUID_LONG_EMA_WINDOW", 36)
        self.SQUID_MOMENTUM_THRESHOLD = CONFIG_HYPERPARAMS.get("SQUID_MOMENTUM_THRESHOLD", 10)

        # Late-day risk adjustments.
        self.LATE_DAY_TIMESTAMP = CONFIG_HYPERPARAMS.get("LATE_DAY_TIMESTAMP", 861778)
        self.LATE_DAY_SIZE_FACTOR = CONFIG_HYPERPARAMS.get("LATE_DAY_SIZE_FACTOR", 0.7066)
        self.LATE_DAY_SPREAD_FACTOR = CONFIG_HYPERPARAMS.get("LATE_DAY_SPREAD_FACTOR", 1.8936)

    # ---------- Common Utility Functions ----------
    def midprice(self, depth: OrderDepth) -> Optional[float]:
        """
        Compute the midprice as the average of the best bid and best ask.
        Returns None if either side is missing.
        """
        if not depth.buy_orders or not depth.sell_orders:
            return None
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        return (best_bid + best_ask) / 2

    def update_rolling_history(self, trader_state: dict, key: str, value: float) -> List[float]:
        """
        Update the rolling history for a given key with the new value.
        Keeps the history length at most self.rolling_window.
        """
        history = trader_state.setdefault(key, [])
        history.append(value)
        if len(history) > self.rolling_window:
            history.pop(0)
        return history

    # ---------- Compute Implied Basket Prices ----------
    def compute_picnic_basket_implied_prices(self, state: TradingState, basket: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Compute the implied bid and ask for a picnic basket from underlying components.
        
        For PICNIC_BASKET1: 6 * CROISSANTS + 3 * JAMS + 1 * DJEMBES
        For PICNIC_BASKET2: 4 * CROISSANTS + 2 * JAMS
        
        Returns:
            A tuple (implied_bid, implied_ask). If any component data is missing, returns (None, None).
        """
        if basket == self.product_basket1:
            weights = {self.product_croissants: 6, self.product_jams: 3, self.product_djembes: 1}
        elif basket == self.product_basket2:
            weights = {self.product_croissants: 4, self.product_jams: 2}
        else:
            return None, None

        total_bid = 0
        total_ask = 0
        for comp, multiplier in weights.items():
            if comp not in state.order_depths:
                return None, None
            comp_depth = state.order_depths[comp]
            if not comp_depth.buy_orders or not comp_depth.sell_orders:
                return None, None
            comp_best_bid = max(comp_depth.buy_orders.keys())
            comp_best_ask = min(comp_depth.sell_orders.keys())
            total_bid += multiplier * comp_best_bid
            total_ask += multiplier * comp_best_ask
        return total_bid, total_ask

    # ---------- Modified Basket Arbitrage Function ----------
    def execute_basket_arbitrage_implied(self, basket_product: str, state: TradingState,
                                        trader_state: dict, spread_history_key: str,
                                        basket_pos_limit: int) -> List[Order]:
        """
        Execute basket arbitrage using implied bid/ask prices from components.
        Compares the market's best bid/ask for the basket with the implied prices.
        If the spreads exceed the threshold, orders are sent to arbitrage;
        otherwise, fallback market making orders are issued.
        """
        orders: List[Order] = []
        basket_depth = state.order_depths[basket_product]
        market_bid = max(basket_depth.buy_orders.keys()) if basket_depth.buy_orders else None
        market_ask = min(basket_depth.sell_orders.keys()) if basket_depth.sell_orders else None

        if market_bid is None or market_ask is None:
            mid = self.midprice(basket_depth) or 0
            return self.create_market_making_orders(basket_product, mid,
                                                    state.position.get(basket_product, 0),
                                                    basket_pos_limit, 1)

        implied_bid, implied_ask = self.compute_picnic_basket_implied_prices(state, basket_product)
        if implied_bid is None or implied_ask is None:
            mid = self.midprice(basket_depth) or 0
            return self.create_market_making_orders(basket_product, mid,
                                                    state.position.get(basket_product, 0),
                                                    basket_pos_limit, 1)

        spread_bid = market_bid - implied_bid
        spread_ask = market_ask - implied_ask

        # Update spread history (this can be used for volatility measurements)
        self.update_rolling_history(trader_state, spread_history_key, (spread_bid + spread_ask) / 2)
        threshold = self.zscore_threshold

        pos_basket = state.position.get(basket_product, 0)
        # If market bid exceeds implied bid by more than threshold, sell basket and buy components.
        if spread_bid > threshold:
            qty = min(ARBITRAGE_QUANTITY, basket_pos_limit + pos_basket)
            if qty > 0:
                orders.append(Order(basket_product, market_bid, -qty))
                recipe = ([(self.product_croissants, 6), (self.product_jams, 3), (self.product_djembes, 1)]
                        if basket_product == self.product_basket1
                        else [(self.product_croissants, 4), (self.product_jams, 2)])
                for comp, multiplier in recipe:
                    comp_depth = state.order_depths.get(comp)
                    if comp_depth and comp_depth.sell_orders:
                        comp_best_ask = min(comp_depth.sell_orders.keys())
                        orders.append(Order(comp, comp_best_ask, multiplier * qty))
            return orders

        # If market ask is below implied ask by more than threshold, buy basket and sell components.
        if spread_ask < -threshold:
            qty = min(ARBITRAGE_QUANTITY, basket_pos_limit - pos_basket)
            if qty > 0:
                orders.append(Order(basket_product, market_ask, qty))
                recipe = ([(self.product_croissants, 6), (self.product_jams, 3), (self.product_djembes, 1)]
                        if basket_product == self.product_basket1
                        else [(self.product_croissants, 4), (self.product_jams, 2)])
                for comp, multiplier in recipe:
                    comp_depth = state.order_depths.get(comp)
                    if comp_depth and comp_depth.buy_orders:
                        comp_best_bid = max(comp_depth.buy_orders.keys())
                        orders.append(Order(comp, comp_best_bid, -multiplier * qty))
            return orders

        # Otherwise, no strong arbitrage signal; fallback to generic market making.
        mid = self.midprice(basket_depth) or 0
        return self.create_market_making_orders(basket_product, mid,
                                                pos_basket, basket_pos_limit, 1)

    # ---------- Fallback Market Making Function ----------
    def create_market_making_orders(self, product: str, fv: float, current_position: int,
                                    position_limit: int, spread: float) -> List[Order]:
        """
        Generate generic market-making orders around a given fair value.
        Buys at (fv - spread) if under long limit and sells at (fv + spread) if under short limit.
        """
        orders = []
        if current_position < position_limit:
            qty = position_limit - current_position
            orders.append(Order(product, int(round(fv - spread)), qty))
        if current_position > -position_limit:
            qty = position_limit + current_position
            orders.append(Order(product, int(round(fv + spread)), -qty))
        return orders

    def do_nothing(self, *args, **kwargs):
        return []
    # ---------- New Helper: Component Market Making Orders ----------
    def create_component_market_making_orders(self, product: str, depth: OrderDepth,
                                            current_position: int, position_limit: int) -> List[Order]:
        """
        Generate market-making orders for a basket component using its best executable prices.
        For a buy order, uses the best ask price; for a sell order, uses the best bid price.
        """
        orders = []
        # For buying: use best ask.
        if depth.sell_orders:
            best_ask = min(depth.sell_orders.keys())
            available_buy = -depth.sell_orders[best_ask]
            buy_qty = min(position_limit - current_position, available_buy)
            if buy_qty >= MIN_ORDER_VOLUME:
                orders.append(Order(product, best_ask, buy_qty))
        # For selling: use best bid.
        if depth.buy_orders:
            best_bid = max(depth.buy_orders.keys())
            available_sell = depth.buy_orders[best_bid]
            sell_qty = min(position_limit + current_position, available_sell)
            if sell_qty >= MIN_ORDER_VOLUME:
                orders.append(Order(product, best_bid, -sell_qty))
        return orders

    def create_component_market_making_orders_v1(self, product: str, depth: OrderDepth,
        current_position: int,
        position_limit: int) -> List[Order]:
        """
        Generate market-making orders for an individual basket component with safeguards.


        Collapse
        - When buying, it uses the best ask price and limits the order size to the smaller 
        of the available volume and the gap to the long position limit, capped by MAX_SAFE_ORDER_SIZE.
        - When selling, it uses the best bid price and limits the order size to the smaller 
        of the available volume and the gap to the short position limit, similarly capped.
        
        Args:
            product: The name of the component product.
            depth: The OrderDepth instance for the component.
            current_position: Your current position in the component.
            position_limit: The absolute limit for the component.
            
        Returns:
            A list of Order objects for this component.
        """
        orders = []

        # For buying: use best ask (lowest sell price)
        if depth.sell_orders:
            best_ask = min(depth.sell_orders.keys())
            available_buy = -depth.sell_orders[best_ask]  # sell_orders have negative volume
            # How many units can be bought without exceeding long position
            potential_qty = position_limit - current_position
            buy_qty = min(available_buy, potential_qty, MAX_SAFE_ORDER_SIZE)
            if buy_qty >= 1:  # use a threshold (e.g. 1 unit) to avoid negligible orders
                orders.append(Order(product, best_ask, buy_qty))

        # For selling: use best bid (highest buy price)
        if depth.buy_orders:
            best_bid = max(depth.buy_orders.keys())
            available_sell = depth.buy_orders[best_bid]
            potential_qty = position_limit + current_position  # for short, allowable sell quantity
            sell_qty = min(available_sell, potential_qty, MAX_SAFE_ORDER_SIZE)
            if sell_qty >= 1:
                orders.append(Order(product, best_bid, -sell_qty))

        return orders
    # ---------- Process Functions for Individual Strategies ----------
    def process_basket_arbitrage(self, state: TradingState, trader_state: dict,
                                result: Dict[str, List[Order]]) -> None:
        """
        Process arbitrage for picnic baskets using the implied bid/ask approach.
        Executes arbitrage for both PICNIC_BASKET1 and PICNIC_BASKET2.
        """
        if self.product_basket1 in state.order_depths:
            orders = self.execute_basket_arbitrage_implied(
                basket_product=self.product_basket1,
                state=state, trader_state=trader_state,
                spread_history_key="spread_history_basket1",
                basket_pos_limit=self.POS_LIMIT_BASKET1
            )
            result[self.product_basket1] = orders
        if self.product_basket2 in state.order_depths:
            orders = self.execute_basket_arbitrage_implied(
                basket_product=self.product_basket2,
                state=state, trader_state=trader_state,
                spread_history_key="spread_history_basket2",
                basket_pos_limit=self.POS_LIMIT_BASKET2  # Using separate limit for basket2
            )
            result[self.product_basket2] = orders

    def process_basket_components(self, state: TradingState,
                                result: Dict[str, List[Order]]) -> None:
        """
        Generate fallback market making orders for each basket component
        (CROISSANTS, JAMS, DJEMBES) using their best executable prices.
        """
        for prod, pos_limit in [(self.product_croissants, self.POS_LIMIT_CROISSANTS),
                                (self.product_jams, self.POS_LIMIT_JAMS),
                                (self.product_djembes, self.POS_LIMIT_DJEMBES)]:
            if prod in state.order_depths:
                depth = state.order_depths[prod]
                pos = state.position.get(prod, 0)
                orders = self.create_component_market_making_orders_v1(prod, depth, pos, pos_limit)
                if orders:
                    result[prod] = orders

    def process_resin(self, state: TradingState, trader_state: dict,
                    result: Dict[str, List[Order]], late_day_factor: float) -> None:
        """
        Process orders for Rainforest Resin using a fixed fair value strategy.
        """
        resin = self.product_resin
        if resin not in state.order_depths:
            return
        resin_depth: OrderDepth = state.order_depths[resin]
        orders: List[Order] = []
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
                    orders.append(Order(resin, round(best_ask), qty))
        if resin_depth.buy_orders:
            best_bid = max(resin_depth.buy_orders.keys())
            vol = resin_depth.buy_orders[best_bid]
            if best_bid >= fv + effective_take_width:
                qty = min(vol, int((self.POSITION_LIMIT + current_position) * size_multiplier))
                if qty > 0:
                    orders.append(Order(resin, round(best_bid), -qty))
        if current_position < self.POSITION_LIMIT:
            qty = int((self.POSITION_LIMIT - current_position) * size_multiplier)
            orders.append(Order(resin, round(fv - effective_mm_edge), qty))
        if current_position > -self.POSITION_LIMIT:
            qty = int((self.POSITION_LIMIT + current_position) * size_multiplier)
            orders.append(Order(resin, round(fv + effective_mm_edge), -qty))
        result[resin] = orders

    def process_kelp(self, state: TradingState, trader_state: dict,
                    result: Dict[str, List[Order]], late_day_factor: float) -> None:
        """
        Process orders for Kelp using a dynamic fair value computed with an EMA.
        """
        kelp = self.product_kelp
        if kelp not in state.order_depths:
            return
        kelp_depth: OrderDepth = state.order_depths[kelp]
        orders: List[Order] = []
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
                    orders.append(Order(kelp, best_ask, qty))
        if kelp_depth.buy_orders:
            best_bid = max(kelp_depth.buy_orders.keys())
            vol = kelp_depth.buy_orders[best_bid]
            if best_bid >= fv + effective_take_width:
                qty = min(vol, int((self.POSITION_LIMIT + current_position) * size_multiplier))
                if qty > 0:
                    orders.append(Order(kelp, best_bid, -qty))
        if current_position < self.POSITION_LIMIT:
            qty = int((self.POSITION_LIMIT - current_position) * size_multiplier)
            orders.append(Order(kelp, int(round(fv - effective_mm_edge)), qty))
        if current_position > -self.POSITION_LIMIT:
            qty = int((self.POSITION_LIMIT + current_position) * size_multiplier)
            orders.append(Order(kelp, int(round(fv + effective_mm_edge)), -qty))
        result[kelp] = orders

    def process_squid(self, state: TradingState, trader_state: dict,
                    result: Dict[str, List[Order]]) -> None:
        """
        Process orders for Squid Ink using a momentum strategy.
        """
        squid = self.product_squid
        if squid not in state.order_depths:
            return
        squid_depth: OrderDepth = state.order_depths[squid]
        orders: List[Order] = []
        current_position = state.position.get(squid, 0)
        if squid_depth.buy_orders and squid_depth.sell_orders:
            best_bid = max(squid_depth.buy_orders.keys())
            best_ask = min(squid_depth.sell_orders.keys())
            current_mid = (best_bid + best_ask) / 2
        else:
            current_mid = 5000
        momentum = self.compute_squid_momentum(squid_depth, trader_state)
        if momentum > self.SQUID_MOMENTUM_THRESHOLD:
            if squid_depth.sell_orders:
                best_ask = min(squid_depth.sell_orders.keys())
                ask_vol = -squid_depth.sell_orders[best_ask]
                qty = min(ask_vol, self.POSITION_LIMIT - current_position)
                if qty > 0:
                    orders.append(Order(squid, best_ask, qty))
        elif momentum < -self.SQUID_MOMENTUM_THRESHOLD:
            if squid_depth.buy_orders:
                best_bid = max(squid_depth.buy_orders.keys())
                bid_vol = squid_depth.buy_orders[best_bid]
                qty = min(bid_vol, self.POSITION_LIMIT + current_position)
                if qty > 0:
                    orders.append(Order(squid, best_bid, -qty))
        if current_position < self.POSITION_LIMIT:
            buy_qty = self.POSITION_LIMIT - current_position
            orders.append(Order(squid, int(round(current_mid - 1)), buy_qty))
        if current_position > -self.POSITION_LIMIT:
            sell_qty = self.POSITION_LIMIT + current_position
            orders.append(Order(squid, int(round(current_mid + 1)), -sell_qty))
        result[squid] = orders

    def compute_kelp_fair_value(self, depth: OrderDepth, trader_state: dict) -> float:
        """
        Compute a dynamic fair value for Kelp using an EMA on the midprice.
        """
        if not depth.buy_orders or not depth.sell_orders:
            return trader_state.get("kelp_last_price", 10000)
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        observed_mid = (best_bid + best_ask) / 2
        last_price = trader_state.get("kelp_last_price", observed_mid)
        new_fv = last_price + self.KELP_BETA * (observed_mid - last_price)
        trader_state["kelp_last_price"] = new_fv
        return new_fv

    def compute_squid_momentum(self, depth: OrderDepth, trader_state: dict) -> float:
        """
        Compute the momentum for Squid Ink as the difference between the current midprice and previous midprice.
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

    def get_late_day_factor(self, timestamp: int) -> float:
        """
        Return the late-day size/spread factor based on the current timestamp.
        """
        return self.LATE_DAY_SIZE_FACTOR if timestamp >= self.LATE_DAY_TIMESTAMP else 1.0

    # ---------- Combined Run Method ----------
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, Any]:
        """
        Main run method.
        Loads persistent trader state, processes each asset strategy,
        and returns:
        - A dictionary mapping product names to orders,
        - Conversion requests (currently 0),
        - Encoded trader state for persistence.
        """
        trader_state: dict = {}
        if state.traderData and state.traderData != "":
            trader_state = jsonpickle.decode(state.traderData)
        result: Dict[str, List[Order]] = {}
        late_day_factor = self.get_late_day_factor(state.timestamp)

        self.process_basket_arbitrage(state, trader_state, result)
        self.process_basket_components(state, result)
        self.process_resin(state, trader_state, result, late_day_factor)
        self.process_kelp(state, trader_state, result, late_day_factor)
        self.process_squid(state, trader_state, result)

        traderData = jsonpickle.encode(trader_state)
        conversions = 0  # Placeholder for future conversion handling.
        return result, conversions, traderData