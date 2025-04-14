from datamodel import Order, TradingState, OrderDepth
import jsonpickle


class Trader:
    def __init__(self):
        best_hyperparams = {
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
        # General
        self.POSITION_LIMIT = 50

        # Late-day behavior
        self.LATE_DAY_TIMESTAMP = best_hyperparams['LATE_DAY_TIMESTAMP']
        self.LATE_DAY_SIZE_FACTOR = best_hyperparams['LATE_DAY_SIZE_FACTOR']
        self.LATE_DAY_SPREAD_FACTOR = best_hyperparams['LATE_DAY_SPREAD_FACTOR']

        # Resin
        self.RESIN_FAIR_VALUE = 10000
        self.RESIN_TAKE_WIDTH = best_hyperparams['RESIN_TAKE_WIDTH']
        self.RESIN_MM_EDGE = best_hyperparams['RESIN_MM_EDGE']

        # Kelp
        self.KELP_TAKE_WIDTH = best_hyperparams['KELP_TAKE_WIDTH']
        self.KELP_MM_EDGE = best_hyperparams['KELP_MM_EDGE']
        self.KELP_BETA = best_hyperparams['KELP_BETA']

        # Squid Ink
        self.SQUID_TAKE_WIDTH = best_hyperparams['SQUID_TAKE_WIDTH']
        self.SQUID_MM_EDGE = best_hyperparams['SQUID_MM_EDGE']
        self.SQUID_BETA = best_hyperparams['SQUID_BETA']
        self.SQUID_TREND_THRESHOLD = best_hyperparams['SQUID_TREND_THRESHOLD']
        self.SQUID_TREND_BIAS = best_hyperparams['SQUID_TREND_BIAS']
        self.SQUID_VOL_THRESHOLD = best_hyperparams['SQUID_VOL_THRESHOLD']
        self.SQUID_SHORT_EMA_WINDOW = best_hyperparams['SQUID_SHORT_EMA_WINDOW']
        self.SQUID_LONG_EMA_WINDOW = best_hyperparams['SQUID_LONG_EMA_WINDOW']

        self.product_resin = "RAINFOREST_RESIN"
        self.product_kelp = "KELP"
        self.product_squid = "SQUID_INK"

    def update_ema(self, last_ema, price, window):
        alpha = 2 / (window + 1)
        if last_ema is None:
            return price
        return alpha * price + (1 - alpha) * last_ema

    def compute_kelp_fair_value(self, depth: OrderDepth, trader_state: dict) -> float:
        if not depth.buy_orders or not depth.sell_orders:
            return trader_state.get("kelp_last_fair", 10000)
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        observed_mid = (best_bid + best_ask) / 2
        last_fair = trader_state.get("kelp_last_fair", observed_mid)
        new_fair = last_fair + self.KELP_BETA * (observed_mid - last_fair)
        trader_state["kelp_last_fair"] = new_fair
        return new_fair

    def compute_squid_fair_value(self, depth: OrderDepth, trader_state: dict) -> float:
        if not depth.buy_orders or not depth.sell_orders:
            return trader_state.get("squid_last_fair", 5000)
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        observed_mid = (best_bid + best_ask) / 2

        short_ema = trader_state.get("squid_short_ema", observed_mid)
        long_ema = trader_state.get("squid_long_ema", observed_mid)
        short_ema = self.update_ema(short_ema, observed_mid, self.SQUID_SHORT_EMA_WINDOW)
        long_ema = self.update_ema(long_ema, observed_mid, self.SQUID_LONG_EMA_WINDOW)
        trader_state["squid_short_ema"] = short_ema
        trader_state["squid_long_ema"] = long_ema

        last_fair = trader_state.get("squid_last_fair", observed_mid)
        new_fair = (1 - self.SQUID_BETA) * last_fair + self.SQUID_BETA * observed_mid

        trend_strength = short_ema - long_ema
        if trend_strength < -self.SQUID_TREND_THRESHOLD:
            new_fair += self.SQUID_TREND_BIAS * trend_strength
        trader_state["squid_last_fair"] = new_fair
        trader_state["squid_last_mid"] = observed_mid
        return new_fair

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

    def get_late_day_factor(self, timestamp: int) -> float:
        return self.LATE_DAY_SIZE_FACTOR if timestamp >= self.LATE_DAY_TIMESTAMP else 1.0

    def run(self, state: TradingState):
        trader_state = jsonpickle.decode(state.traderData) if state.traderData else {}

        result = {}
        late_day_factor = self.get_late_day_factor(state.timestamp)

        # Resin
        if self.product_resin in state.order_depths:
            depth = state.order_depths[self.product_resin]
            orders = []
            pos = state.position.get(self.product_resin, 0)
            fv = self.RESIN_FAIR_VALUE
            take_width = self.RESIN_TAKE_WIDTH * (self.LATE_DAY_SPREAD_FACTOR if late_day_factor != 1.0 else 1)
            mm_edge = self.RESIN_MM_EDGE * (self.LATE_DAY_SPREAD_FACTOR if late_day_factor != 1.0 else 1)
            size_mult = late_day_factor

            if depth.sell_orders:
                best_ask = min(depth.sell_orders.keys())
                vol = -depth.sell_orders[best_ask]
                if best_ask <= fv - take_width:
                    qty = min(vol, int((self.POSITION_LIMIT - pos) * size_mult))
                    if qty > 0:
                        orders.append(Order(self.product_resin, best_ask, qty))
            if depth.buy_orders:
                best_bid = max(depth.buy_orders.keys())
                vol = depth.buy_orders[best_bid]
                if best_bid >= fv + take_width:
                    qty = min(vol, int((self.POSITION_LIMIT + pos) * size_mult))
                    if qty > 0:
                        orders.append(Order(self.product_resin, best_bid, -qty))
            if pos < self.POSITION_LIMIT:
                qty = int((self.POSITION_LIMIT - pos) * size_mult)
                orders.append(Order(self.product_resin, fv - mm_edge, qty))
            if pos > -self.POSITION_LIMIT:
                qty = int((self.POSITION_LIMIT + pos) * size_mult)
                orders.append(Order(self.product_resin, fv + mm_edge, -qty))
            result[self.product_resin] = orders

        # Kelp
        if self.product_kelp in state.order_depths:
            depth = state.order_depths[self.product_kelp]
            orders = []
            pos = state.position.get(self.product_kelp, 0)
            fv = self.compute_kelp_fair_value(depth, trader_state)
            take_width = self.KELP_TAKE_WIDTH * (self.LATE_DAY_SPREAD_FACTOR if late_day_factor != 1.0 else 1)
            mm_edge = self.KELP_MM_EDGE * (self.LATE_DAY_SPREAD_FACTOR if late_day_factor != 1.0 else 1)
            size_mult = late_day_factor

            if depth.sell_orders:
                best_ask = min(depth.sell_orders.keys())
                vol = -depth.sell_orders[best_ask]
                if best_ask <= fv - take_width:
                    qty = min(vol, int((self.POSITION_LIMIT - pos) * size_mult))
                    if qty > 0:
                        orders.append(Order(self.product_kelp, best_ask, qty))
            if depth.buy_orders:
                best_bid = max(depth.buy_orders.keys())
                vol = depth.buy_orders[best_bid]
                if best_bid >= fv + take_width:
                    qty = min(vol, int((self.POSITION_LIMIT + pos) * size_mult))
                    if qty > 0:
                        orders.append(Order(self.product_kelp, best_bid, -qty))
            if pos < self.POSITION_LIMIT:
                qty = int((self.POSITION_LIMIT - pos) * size_mult)
                orders.append(Order(self.product_kelp, round(fv - mm_edge), qty))
            if pos > -self.POSITION_LIMIT:
                qty = int((self.POSITION_LIMIT + pos) * size_mult)
                orders.append(Order(self.product_kelp, round(fv + mm_edge), -qty))
            result[self.product_kelp] = orders

        # Squid Ink
        if self.product_squid in state.order_depths:
            depth = state.order_depths[self.product_squid]
            orders = []
            pos = state.position.get(self.product_squid, 0)
            fv = self.compute_squid_fair_value(depth, trader_state)
            vol = self.compute_squid_volatility(depth, trader_state)
            take_width = self.SQUID_TAKE_WIDTH * (self.LATE_DAY_SPREAD_FACTOR if late_day_factor != 1.0 else 1)
            mm_edge = self.SQUID_MM_EDGE * (self.LATE_DAY_SPREAD_FACTOR if late_day_factor != 1.0 else 1)
            size_mult = late_day_factor

            if depth.sell_orders:
                best_ask = min(depth.sell_orders.keys())
                vol_ask = -depth.sell_orders[best_ask]
                if best_ask <= fv - take_width and vol_ask <= self.SQUID_VOL_THRESHOLD:
                    qty = min(vol_ask, int((self.POSITION_LIMIT - pos) * size_mult))
                    if qty > 0:
                        orders.append(Order(self.product_squid, best_ask, qty))
            if depth.buy_orders:
                best_bid = max(depth.buy_orders.keys())
                vol_bid = depth.buy_orders[best_bid]
                if best_bid >= fv + take_width and vol_bid <= self.SQUID_VOL_THRESHOLD:
                    qty = min(vol_bid, int((self.POSITION_LIMIT + pos) * size_mult))
                    if qty > 0:
                        orders.append(Order(self.product_squid, best_bid, -qty))
            if pos < self.POSITION_LIMIT:
                qty = int((self.POSITION_LIMIT - pos) * size_mult)
                orders.append(Order(self.product_squid, round(fv - mm_edge), qty))
            if pos > -self.POSITION_LIMIT:
                qty = int((self.POSITION_LIMIT + pos) * size_mult)
                orders.append(Order(self.product_squid, round(fv + mm_edge), -qty))

            short_ema = trader_state.get("squid_short_ema", fv)
            long_ema = trader_state.get("squid_long_ema", fv)
            if state.timestamp >= self.LATE_DAY_TIMESTAMP and (short_ema - long_ema) < -self.SQUID_TREND_THRESHOLD:
                if pos > 0:
                    orders.append(Order(self.product_squid, round(fv - 0.5 * mm_edge), -pos))
            result[self.product_squid] = orders

        traderData = jsonpickle.encode(trader_state)
        conversions = 0
        return result, conversions, traderData
