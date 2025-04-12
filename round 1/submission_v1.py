from datamodel import Order, TradingState, OrderDepth
from typing import List

class Trader:
    def __init__(self):
        # Fixed parameters for Rainforest Resin (position limit 50; fair value 10,000)
        self.product = "RAINFOREST_RESIN"
        self.fair_value = 10000
        self.position_limit = 50
        self.take_width = 5    # How far price must deviate from fair value to immediately take liquidity
        self.mm_edge = 2       # For basic market-making orders (post orders just inside fair value)

    def run(self, state: TradingState):
        # Retrieve the order depth for Rainforest Resin from the TradingState
        product = self.product
        if product not in state.order_depths:
            return {}, 0, ""

        order_depth: OrderDepth = state.order_depths[product]
        orders: List[Order] = []
        current_position = state.position.get(product, 0)

        # -------------------------
        # Step 1: Immediate Order Taking
        # -------------------------
        # If there are sell orders (asks) below our fair value minus the take_width, buy from them.
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_volume = -order_depth.sell_orders[best_ask]  # volume as positive number
            # Check if the price is sufficiently low compared to fair value.
            if best_ask <= self.fair_value - self.take_width:
                # Ensure we do not exceed our long position limit.
                quantity_to_buy = min(best_ask_volume, self.position_limit - current_position)
                if quantity_to_buy > 0:
                    orders.append(Order(product, best_ask, quantity_to_buy))
                    # Optionally, we could update order_depth here to simulate order fills.

        # If there are buy orders (bids) above our fair value plus the take_width, sell into them.
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_volume = order_depth.buy_orders[best_bid]  # volume as positive number
            if best_bid >= self.fair_value + self.take_width:
                # Ensure we do not exceed our short position limit.
                quantity_to_sell = min(best_bid_volume, self.position_limit + current_position)
                if quantity_to_sell > 0:
                    orders.append(Order(product, best_bid, -quantity_to_sell))
                    # Optionally update order_depth for a simulated fill.

        # -------------------------
        # Step 2: Market-Making
        # -------------------------
        # If there is remaining room to increase or decrease our position, post market-making orders.
        # Post a buy order just below fair value if we are not at the long limit.
        if current_position < self.position_limit:
            buy_order_qty = self.position_limit - current_position
            orders.append(Order(product, self.fair_value - self.mm_edge, buy_order_qty))
        
        # Post a sell order just above fair value if we have a short room.
        if current_position > -self.position_limit:
            sell_order_qty = self.position_limit + current_position  # Note: current_position can be negative
            orders.append(Order(product, self.fair_value + self.mm_edge, -sell_order_qty))

        # -------------------------
        # Prepare Return Values
        # -------------------------
        # The algorithm’s traderData can be a string; here we leave it unchanged.
        traderData = ""
        conversions = 0  # No conversion logic for Round 1 in this simple strategy
        
        return {product: orders}, conversions, traderData
