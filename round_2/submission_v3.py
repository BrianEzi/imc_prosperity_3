import jsonpickle
from datamodel import Order, TradingState, OrderDepth
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

#--- Constants for Basket Arbitrage ---
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
        """Initialize parameters and hyperparameters for strategies."""
        # --- Basket & Arbitrage Product Names ---
        self.product_croissants = "CROISSANTS"
        self.product_jams = "JAMS"
        self.product_djembes = "DJEMBES"
        self.product_basket1 = "PICNIC_BASKET1"
        # --- Other Assets ---
        self.product_resin = "RAINFOREST_RESIN"
        self.product_kelp = "KELP"
        self.product_squid = "SQUID_INK"

        # --- Position Limits ---
        self.POS_LIMIT_BASKET1 = POSITION_LIMIT_BASKET1
        self.POS_LIMIT_CROISSANTS = POSITION_LIMIT_CROISSANTS
        self.POS_LIMIT_JAMS = POSITION_LIMIT_JAMS
        self.POS_LIMIT_DJEMBES = POSITION_LIMIT_DJEMBES
        # For individual assets (for resin, kelp, squid)
        self.POSITION_LIMIT = 50

        # --- Basket Arbitrage Hyperparameters ---
        self.default_spread_mean = DEFAULT_SPREAD_MEAN
        self.rolling_window = ROLLING_WINDOW
        self.zscore_threshold = ZSCORE_THRESHOLD

        # --- Hyperparameters for Resin, Kelp, and Squid ---
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
        # Resin parameters (fixed fair value)
        self.RESIN_FAIR_VALUE = 10000
        self.RESIN_TAKE_WIDTH = HYPERPARAMS['RESIN_TAKE_WIDTH']
        self.RESIN_MM_EDGE = HYPERPARAMS['RESIN_MM_EDGE']

        # Kelp parameters (dynamic fair value)
        self.KELP_TAKE_WIDTH = HYPERPARAMS['KELP_TAKE_WIDTH']
        self.KELP_MM_EDGE = HYPERPARAMS['KELP_MM_EDGE']
        self.KELP_BETA = HYPERPARAMS['KELP_BETA']

        # Squid parameters (momentum strategy)
        self.SQUID_TAKE_WIDTH = HYPERPARAMS['SQUID_TAKE_WIDTH']
        self.SQUID_MM_EDGE = HYPERPARAMS['SQUID_MM_EDGE']
        self.SQUID_BETA = HYPERPARAMS['SQUID_BETA']
        self.SQUID_TREND_THRESHOLD = HYPERPARAMS['SQUID_TREND_THRESHOLD']
        self.SQUID_TREND_BIAS = HYPERPARAMS['SQUID_TREND_BIAS']
        self.SQUID_VOL_THRESHOLD = HYPERPARAMS['SQUID_VOL_THRESHOLD']
        self.SQUID_SHORT_EMA_WINDOW = HYPERPARAMS['SQUID_SHORT_EMA_WINDOW']
        self.SQUID_LONG_EMA_WINDOW = HYPERPARAMS['SQUID_LONG_EMA_WINDOW']
        self.SQUID_MOMENTUM_THRESHOLD = 10  # tunable threshold

        # Late-day risk adjustments
        self.LATE_DAY_TIMESTAMP = HYPERPARAMS['LATE_DAY_TIMESTAMP']
        self.LATE_DAY_SIZE_FACTOR = HYPERPARAMS['LATE_DAY_SIZE_FACTOR']
        self.LATE_DAY_SPREAD_FACTOR = HYPERPARAMS['LATE_DAY_SPREAD_FACTOR']

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

    def update_rolling_history(self, trader_state: dict, key: str,
                            value: float) -> List[float]:
        """
        Update the rolling history list for a given key with the new value.
        Pops the oldest element if the list exceeds the rolling window.
        """
        history = trader_state.setdefault(key, [])
        history.append(value)
        if len(history) > self.rolling_window:
            history.pop(0)
        return history

    # ---------- Basket Fair Value Functions ----------
    def compute_component_fv(self, product: str,
                            state: TradingState) -> Optional[float]:
        """
        Compute midprice (fair value) for an asset using its order depth.
        """
        if product not in state.order_depths:
            return None
        return self.midprice(state.order_depths[product])

    def compute_basket_theoretical_value(
            self, state: TradingState) -> Tuple[Optional[float], Optional[float]]:
        """
        Compute theoretical basket values based on the recipes:
            PICNIC_BASKET1 = 6 * CROISSANTS + 3 * JAMS + 1 * DJEMBES
            PICNIC_BASKET2 = 4 * CROISSANTS + 2 * JAMS
        """
        fv_c = self.compute_component_fv(self.product_croissants, state)
        fv_j = self.compute_component_fv(self.product_jams, state)
        fv_d = self.compute_component_fv(self.product_djembes, state)
        basket1 = None
        basket2 = None
        if fv_c is not None and fv_j is not None and fv_d is not None:
            basket1 = 6 * fv_c + 3 * fv_j + fv_d
        if fv_c is not None and fv_j is not None:
            basket2 = 4 * fv_c + 2 * fv_j
        return basket1, basket2

    def compute_basket2_theoretical_value(self,
                                        state: TradingState) -> Optional[float]:
        """
        Compute the theoretical value for PICNIC_BASKET2:
            PICNIC_BASKET2 = 4 * CROISSANTS + 2 * JAMS
        Returns None if any component midprice is unavailable.
        """
        fv_c = self.compute_component_fv(self.product_croissants, state)
        fv_j = self.compute_component_fv(self.product_jams, state)
        if fv_c is None or fv_j is None:
            return None
        return 4 * fv_c + 2 * fv_j

    # ---------- Generic Basket Arbitrage Function ----------
    def execute_basket_arbitrage_generic(
            self, basket_product: str, basket_mid: float,
            theoretical_value: float, component_recipe: List[Tuple[str, int]],
            state: TradingState, trader_state: dict,
            spread_history_key: str, basket_pos_limit: int) -> List[Order]:
        """
        Execute a generic basket arbitrage strategy.
        Depending on the spread z-score, the function either generates
        arbitrage orders (selling or buying the basket and trading the
        underlying components) or defaults to market making.
        """
        orders: List[Order] = []
        spread = basket_mid - theoretical_value
        history = self.update_rolling_history(trader_state, spread_history_key, spread)
        if len(history) < self.rolling_window:
            return self.create_market_making_orders(
                basket_product, basket_mid,
                state.position.get(basket_product, 0),
                basket_pos_limit, 1
            )

        rolling_std = np.std(history)
        zscore = ((spread - self.default_spread_mean) / rolling_std
                if rolling_std > 0 else 0.0)
        pos_basket = state.position.get(basket_product, 0)

        if zscore > self.zscore_threshold:
            # Basket is overpriced: sell basket, buy underlying components.
            best_bid = (max(state.order_depths[basket_product].buy_orders.keys())
                        if state.order_depths[basket_product].buy_orders
                        else basket_mid)
            qty = min(ARBITRAGE_QUANTITY, basket_pos_limit + pos_basket)
            if qty > 0:
                orders.append(Order(basket_product, best_bid, -qty))
                for comp, multiplier in component_recipe:
                    if comp in state.order_depths:
                        comp_fv = self.compute_component_fv(comp, state)
                        if comp_fv is not None:
                            comp_depth = state.order_depths[comp]
                            best_ask = (min(comp_depth.sell_orders.keys())
                                        if comp_depth.sell_orders else comp_fv)
                            orders.append(Order(comp, best_ask, multiplier * qty))
        elif zscore < -self.zscore_threshold:
            # Basket is underpriced: buy basket, sell underlying components.
            best_ask = (min(state.order_depths[basket_product].sell_orders.keys())
                        if state.order_depths[basket_product].sell_orders
                        else basket_mid)
            qty = min(ARBITRAGE_QUANTITY, basket_pos_limit - pos_basket)
            if qty > 0:
                orders.append(Order(basket_product, best_ask, qty))
                for comp, multiplier in component_recipe:
                    if comp in state.order_depths:
                        comp_fv = self.compute_component_fv(comp, state)
                        if comp_fv is not None:
                            comp_depth = state.order_depths[comp]
                            best_bid = (max(comp_depth.buy_orders.keys())
                                        if comp_depth.buy_orders else comp_fv)
                            orders.append(Order(comp, best_bid, -multiplier * qty))
        else:
            # No strong signal: fallback to market making.
            orders = self.create_market_making_orders(
                basket_product, basket_mid, pos_basket, basket_pos_limit, 1
            )
        return orders

    def create_market_making_orders(self, product: str, fv: float,
                                    current_position: int,
                                    position_limit: int, spread: float) -> List[Order]:
        """
        Generate market-making orders around the fair value.
        Buys at (fv - spread) if under long limit and sells at (fv + spread)
        if under short limit.
        """
        orders = []
        if current_position < position_limit:
            qty = position_limit - current_position
            orders.append(Order(product, int(round(fv - spread)), qty))
        if current_position > -position_limit:
            qty = position_limit + current_position
            orders.append(Order(product, int(round(fv + spread)), -qty))
        return orders

    # ---------- Utility Functions for Kelp and Squid Strategies ----------
    def update_ema(self, last_ema: Optional[float], price: float,
                window: int) -> float:
        """
        Update and return the exponential moving average.
        Uses the formula with alpha = 2/(window+1).
        """
        alpha = 2 / (window + 1)
        if last_ema is None:
            return price
        return alpha * price + (1 - alpha) * last_ema

    def compute_kelp_fair_value(self, depth: OrderDepth,
                                trader_state: dict) -> float:
        """
        Compute the dynamic fair value for Kelp.
        Uses the last price stored in trader_state to adjust the value.
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

    def get_late_day_factor(self, timestamp: int) -> float:
        """
        Return the late-day size/spread factor based on the timestamp.
        """
        return self.LATE_DAY_SIZE_FACTOR if timestamp >= self.LATE_DAY_TIMESTAMP else 1.0

    def compute_squid_momentum(self, depth: OrderDepth,
                            trader_state: dict) -> float:
        """
        Compute the momentum for SQUID_INK as the change in midprice
        from the last tick.
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

    # ---------- Process Functions for Individual Strategies ----------
    def process_basket_arbitrage(self, state: TradingState,
                                trader_state: dict, result: Dict[str, List[Order]]
                                ) -> None:
        """
        Process arbitrage for baskets.
        Uses basket1 and basket2 strategies if order depths are available.
        """
        # Process PICNIC_BASKET1 arbitrage if available.
        if self.product_basket1 in state.order_depths:
            depth = state.order_depths[self.product_basket1]
            basket_mid = self.midprice(depth)
            basket1_theory, _ = self.compute_basket_theoretical_value(state)
            if basket_mid is not None and basket1_theory is not None:
                orders = self.execute_basket_arbitrage_generic(
                    basket_product=self.product_basket1,
                    basket_mid=basket_mid,
                    theoretical_value=basket1_theory,
                    component_recipe=[(self.product_croissants, 6),
                                    (self.product_jams, 3),
                                    (self.product_djembes, 1)],
                    state=state,
                    trader_state=trader_state,
                    spread_history_key="spread_history_basket1",
                    basket_pos_limit=self.POS_LIMIT_BASKET1
                )
                result[self.product_basket1] = orders
            else:
                # Fallback market making if theory/midprice not available.
                result[self.product_basket1] = self.create_market_making_orders(
                    self.product_basket1,
                    basket_mid if basket_mid is not None else 0,
                    state.position.get(self.product_basket1, 0),
                    self.POS_LIMIT_BASKET1, 1
                )
        # Process PICNIC_BASKET2 arbitrage if available.
        basket2 = "PICNIC_BASKET2"
        if basket2 in state.order_depths:
            depth = state.order_depths[basket2]
            basket_mid = self.midprice(depth)
            basket2_theory = self.compute_basket2_theoretical_value(state)
            if basket_mid is not None and basket2_theory is not None:
                orders = self.execute_basket_arbitrage_generic(
                    basket_product=basket2,
                    basket_mid=basket_mid,
                    theoretical_value=basket2_theory,
                    component_recipe=[(self.product_croissants, 4),
                                    (self.product_jams, 2)],
                    state=state,
                    trader_state=trader_state,
                    spread_history_key="spread_history_basket2",
                    basket_pos_limit=self.POS_LIMIT_BASKET1
                )
                result[basket2] = orders

    def process_basket_components(self, state: TradingState,
                                result: Dict[str, List[Order]]) -> None:
        """
        Generate fallback market making orders for the basket components.
        """
        for prod, pos_limit in [(self.product_croissants, self.POS_LIMIT_CROISSANTS),
                                (self.product_jams, self.POS_LIMIT_JAMS),
                                (self.product_djembes, self.POS_LIMIT_DJEMBES)]:
            if prod in state.order_depths:
                depth = state.order_depths[prod]
                fv = self.midprice(depth)
                if fv is None:
                    continue
                pos = state.position.get(prod, 0)
                result[prod] = self.create_market_making_orders(prod, fv, pos, pos_limit, 1)

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
        effective_take_width = (self.RESIN_TAKE_WIDTH *
                                (self.LATE_DAY_SPREAD_FACTOR if late_day_factor != 1.0 else 1))
        effective_mm_edge = (self.RESIN_MM_EDGE *
                            (self.LATE_DAY_SPREAD_FACTOR if late_day_factor != 1.0 else 1))
        size_multiplier = late_day_factor

        # Sell side: if price is attractive for buying (coming from the other side).
        if resin_depth.sell_orders:
            best_ask = min(resin_depth.sell_orders.keys())
            vol = -resin_depth.sell_orders[best_ask]
            if best_ask <= fv - effective_take_width:
                qty = min(vol, int((self.POSITION_LIMIT - current_position) * size_multiplier))
                if qty > 0:
                    orders.append(Order(resin, round(best_ask), qty))
        # Buy side:
        if resin_depth.buy_orders:
            best_bid = max(resin_depth.buy_orders.keys())
            vol = resin_depth.buy_orders[best_bid]
            if best_bid >= fv + effective_take_width:
                qty = min(vol, int((self.POSITION_LIMIT + current_position) * size_multiplier))
                if qty > 0:
                    orders.append(Order(resin, round(best_bid), -qty))
        # Market making on both sides.
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
        Process orders for Kelp using a dynamically computed fair value.
        """
        kelp = self.product_kelp
        if kelp not in state.order_depths:
            return

        kelp_depth: OrderDepth = state.order_depths[kelp]
        orders: List[Order] = []
        current_position = state.position.get(kelp, 0)
        fv = self.compute_kelp_fair_value(kelp_depth, trader_state)
        effective_take_width = (self.KELP_TAKE_WIDTH *
                                (self.LATE_DAY_SPREAD_FACTOR if late_day_factor != 1.0 else 1))
        effective_mm_edge = (self.KELP_MM_EDGE *
                            (self.LATE_DAY_SPREAD_FACTOR if late_day_factor != 1.0 else 1))
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
        Process orders for Squid Ink using a simple momentum strategy.
        """
        squid = self.product_squid
        if squid not in state.order_depths:
            return

        squid_depth: OrderDepth = state.order_depths[squid]
        orders: List[Order] = []
        current_position = state.position.get(squid, 0)

        # Determine current midprice; use fallback if not available.
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
        # Additional fallback market-making orders.
        if current_position < self.POSITION_LIMIT:
            buy_qty = self.POSITION_LIMIT - current_position
            orders.append(Order(squid, int(round(current_mid - 1)), buy_qty))
        if current_position > -self.POSITION_LIMIT:
            sell_qty = self.POSITION_LIMIT + current_position
            orders.append(Order(squid, int(round(current_mid + 1)), -sell_qty))
        result[squid] = orders

    # ---------- Combined Run Method ----------
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, Any]:
        """
        Main run method.
        Loads persistent trader state, processes each asset/strategy,
        and returns the orders, any conversion value, and updated trader data.
        """
        # Load persistent state if present.
        trader_state: dict = {}
        if state.traderData and state.traderData != "":
            trader_state = jsonpickle.decode(state.traderData)

        result: Dict[str, List[Order]] = {}
        late_day_factor = self.get_late_day_factor(state.timestamp)

        # Process basket arbitrage strategies
        self.process_basket_arbitrage(state, trader_state, result)

        # Process fallback market making for basket components
        self.process_basket_components(state, result)

        # Process other individual asset strategies
        self.process_resin(state, trader_state, result, late_day_factor)
        self.process_kelp(state, trader_state, result, late_day_factor)
        self.process_squid(state, trader_state, result)

        traderData = jsonpickle.encode(trader_state)
        conversions = 0  # Placeholder for any conversion handling in the future.
        return result, conversions, traderData