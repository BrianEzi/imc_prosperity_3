import json
from typing import Any

import jsonpickle
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


class Trader:
    def __init__(self):
        # Define product identifiers
        self.product_resin = "RAINFOREST_RESIN"
        self.product_kelp = "KELP"
        self.product_squid = "SQUID_INK"
        self.position_limit = 50

        # Fixed parameters for Rainforest Resin
        self.resin_fair_value = 10000      # Fixed fair value for Rainforest Resin
        self.resin_take_width = 4 #BEST_HYPERPARAMS['resin_take_width']         # Take width for Resin
        self.resin_mm_edge = 2 #BEST_HYPERPARAMS['resin_mm_edge']             # Market-making edge for Resin

        # Parameters for Kelp
        self.kelp_take_width = 8 # BEST_HYPERPARAMS['kelp_take_width']           # Take width for Kelp
        self.kelp_mm_edge = 1.75 # BEST_HYPERPARAMS['kelp_mm_edge']              # Market making edge for Kelp
        self.kelp_beta = 0.6 #  BEST_HYPERPARAMS['kelp_beta']               # Smoothing parameter for dynamic fair value update

        # Parameters for Squid Ink
        self.squid_take_width = 6.8 # BEST_HYPERPARAMS['squid_take_width']          # Take width for Squid Ink (wider due to volatility)
        self.squid_mm_edge = 8.6 #BEST_HYPERPARAMS['squid_mm_edge']             # Market-making edge for Squid Ink
        self.squid_beta = 0.6 # BEST_HYPERPARAMS['squid_beta']              # Mean-reversion parameter for Squid Ink
        self.squid_vol_threshold = 7 # BEST_HYPERPARAMS['squid_vol_threshold']      # Maximum acceptable volume for immediate order taking

    def compute_kelp_fair_value(self, order_depth: OrderDepth, trader_state: dict) -> float:
        """
        Compute a dynamic fair value for Kelp based on the current order book.
        Uses an exponential smoothing update based on the observed midprice.
        """
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
        """
        Compute a dynamic fair value for Squid Ink using a stable EMA update with a capped maximum change.
        Observes the midprice and smooths the update using self.squid_beta. Then, limits the maximum change per iteration.
        """
        # Return a fallback if one side of the order book is missing.
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return trader_state.get("squid_last_price", 5000)  # Fallback initial value

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        
        observed_mid = (best_bid + best_ask) / 2
        last_price = trader_state.get("squid_last_price", observed_mid)
        
        # Use EMA smoothing to update the fair value.
        alpha = self.squid_beta  # e.g., 0.5 typically between 0 and 1
        new_fair = (1 - alpha) * last_price + alpha * observed_mid
        
        # Cap the maximum allowed change per iteration.
        max_change = 1000  # You can tune this value.
        change = new_fair - last_price
        if abs(change) > max_change:
            change = max_change if change > 0 else -max_change
            new_fair = last_price + change

        trader_state["squid_last_price"] = new_fair
        return new_fair

    def run(self, state: TradingState):
        # Load persistent trader state from state.traderData (if any)
        trader_state = {}
        if state.traderData and state.traderData != "":
            trader_state = jsonpickle.decode(state.traderData)

        result = {}

        # --------------------------
        # Process Rainforest Resin (fixed fair value)
        # --------------------------
        resin = self.product_resin
        if resin in state.order_depths:
            order_depth: OrderDepth = state.order_depths[resin]
            orders_resin: list[Order] = []
            current_position = state.position.get(resin, 0)

            if order_depth.sell_orders:
                best_ask = min(order_depth.sell_orders.keys())
                best_ask_volume = -order_depth.sell_orders[best_ask]
                if best_ask <= self.resin_fair_value - self.resin_take_width:
                    quantity = min(best_ask_volume, self.position_limit - current_position)
                    if quantity > 0:
                        orders_resin.append(Order(resin, best_ask, quantity))
            if order_depth.buy_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_bid_volume = order_depth.buy_orders[best_bid]
                if best_bid >= self.resin_fair_value + self.resin_take_width:
                    quantity = min(best_bid_volume, self.position_limit + current_position)
                    if quantity > 0:
                        orders_resin.append(Order(resin, best_bid, -quantity))
            if current_position < self.position_limit:
                buy_qty = self.position_limit - current_position
                orders_resin.append(Order(resin, self.resin_fair_value - self.resin_mm_edge, buy_qty))
            if current_position > -self.position_limit:
                sell_qty = self.position_limit + current_position
                orders_resin.append(Order(resin, self.resin_fair_value + self.resin_mm_edge, -sell_qty))
            result[resin] = orders_resin

        # --------------------------
        # Process Kelp with Dynamic Fair Value
        # --------------------------
        kelp = self.product_kelp
        if kelp in state.order_depths:
            order_depth: OrderDepth = state.order_depths[kelp]
            orders_kelp: list[Order] = []
            current_position = state.position.get(kelp, 0)
            kelp_fair_value = self.compute_kelp_fair_value(order_depth, trader_state)
            if order_depth.sell_orders:
                best_ask = min(order_depth.sell_orders.keys())
                best_ask_volume = -order_depth.sell_orders[best_ask]
                if best_ask <= kelp_fair_value - self.kelp_take_width:
                    quantity = min(best_ask_volume, self.position_limit - current_position)
                    if quantity > 0:
                        orders_kelp.append(Order(kelp, best_ask, quantity))
            if order_depth.buy_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_bid_volume = order_depth.buy_orders[best_bid]
                if best_bid >= kelp_fair_value + self.kelp_take_width:
                    quantity = min(best_bid_volume, self.position_limit + current_position)
                    if quantity > 0:
                        orders_kelp.append(Order(kelp, best_bid, -quantity))
            if current_position < self.position_limit:
                buy_qty = self.position_limit - current_position
                orders_kelp.append(Order(kelp, int(round(kelp_fair_value - self.kelp_mm_edge)), buy_qty))
            if current_position > -self.position_limit:
                sell_qty = self.position_limit + current_position
                orders_kelp.append(Order(kelp, int(round(kelp_fair_value + self.kelp_mm_edge)), -sell_qty))
            result[kelp] = orders_kelp

        # --------------------------
        # Process Squid Ink with Dynamic, Mean-Reverting Fair Value and Volatility Filtering
        # --------------------------
        squid = self.product_squid
        if squid in state.order_depths:
            order_depth: OrderDepth = state.order_depths[squid]
            orders_squid: list[Order] = []
            current_position = state.position.get(squid, 0)
            squid_fair_value = self.compute_squid_fair_value(order_depth, trader_state)
            # Order taking for Squid Ink with volume filtering to avoid adverse orders
            if order_depth.sell_orders:
                best_ask = min(order_depth.sell_orders.keys())
                best_ask_volume = -order_depth.sell_orders[best_ask]
                # Only take the order if volume is below our adverse threshold
                if best_ask <= squid_fair_value - self.squid_take_width and best_ask_volume <= self.squid_vol_threshold:
                    quantity = min(best_ask_volume, self.position_limit - current_position)
                    if quantity > 0:
                        orders_squid.append(Order(squid, best_ask, quantity))
            if order_depth.buy_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_bid_volume = order_depth.buy_orders[best_bid]
                if best_bid >= squid_fair_value + self.squid_take_width and best_bid_volume <= self.squid_vol_threshold:
                    quantity = min(best_bid_volume, self.position_limit + current_position)
                    if quantity > 0:
                        orders_squid.append(Order(squid, best_bid, -quantity))
            # Market-making orders for Squid Ink
            if current_position < self.position_limit:
                buy_qty = self.position_limit - current_position
                orders_squid.append(Order(squid, int(round(squid_fair_value - self.squid_mm_edge)), buy_qty))
            if current_position > -self.position_limit:
                sell_qty = self.position_limit + current_position
                orders_squid.append(Order(squid, int(round(squid_fair_value + self.squid_mm_edge)), -sell_qty))
            result[squid] = orders_squid

        # Persist updated trader state for the next iteration
        traderData = jsonpickle.encode(trader_state)
        conversions = 0

        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData