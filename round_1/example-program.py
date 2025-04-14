from typing import Dict, List
from datamodel import *


class Trader:

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {}

        # Iterate over all the keys (the available products) contained in the order dephts
        for product in state.order_depths.keys():

            # Check if the current product is the 'PEARLS' product, only then run the order logic
            # if product == 'PRODUCT1':

                # Retrieve the Order Depth containing all the market BUY and SELL orders for PEARLS
                order_depth: OrderDepth = state.order_depths[product]

                # Initialize the list of Orders to be sent as an empty list
                orders: list[Order] = []

                # Define a fair value for the PEARLS.
                # Note that this value of 1 is just a dummy value, you should likely change it!
                acceptable_price = 1

                # If statement checks if there are any SELL orders in the PEARLS market
                if len(order_depth.sell_orders) > 0:

                    # Sort all the available sell orders by their price,
                    # and select only the sell order with the lowest price
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_volume = order_depth.sell_orders[best_ask]

                    # Check if the lowest ask (sell order) is lower than the above defined fair value
                    if best_ask < acceptable_price:

                        # In case the lowest ask is lower than our fair value,
                        # This presents an opportunity for us to buy cheaply
                        # The code below therefore sends a BUY order at the price level of the ask,
                        # with the same quantity
                        # We expect this order to trade with the sell order
                        print("BUY", str(-best_ask_volume) + "x", best_ask)
                        orders.append(Order(product, best_ask, -best_ask_volume))

                # The below code block is similar to the one above,
                # the difference is that it find the highest bid (buy order)
                # If the price of the order is higher than the fair value
                # This is an opportunity to sell at a premium
                if len(order_depth.buy_orders) != 0:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_volume = order_depth.buy_orders[best_bid]
                    if best_bid > acceptable_price:
                        print("SELL", str(best_bid_volume) + "x", best_bid)
                        orders.append(Order(product, best_bid, -best_bid_volume))

                # Add all the above the orders to the result dict
                result[product] = orders

                # Return the dict of orders
                # These possibly contain buy or sell orders for PEARLS
                # Depending on the logic above
        return result

if __name__=='__main__':
    timestamp = 1000

    listings = {
        "PRODUCT1": Listing(
            symbol="PRODUCT1", 
            product="PRODUCT1", 
            denomination= "SEASHELLS"
        ),
        "PRODUCT2": Listing(
            symbol="PRODUCT2", 
            product="PRODUCT2", 
            denomination= "SEASHELLS"
        ),
    }

    order_depths = {
        "PRODUCT1": OrderDepth(
            buy_orders={10: 7, 9: 5},
            sell_orders={11: -4, 12: -8}
        ),
        "PRODUCT2": OrderDepth(
            buy_orders={142: 3, 141: 5},
            sell_orders={144: -5, 145: -8}
        ),	
    }

    own_trades = {
        "PRODUCT1": [],
        "PRODUCT2": []
    }

    market_trades = {
        "PRODUCT1": [
            Trade(
                symbol="PRODUCT1",
                price=11,
                quantity=4,
                buyer="",
                seller="",
                timestamp=900
            )
        ],
        "PRODUCT2": []
    }

    position = {
        "PRODUCT1": 3,
        "PRODUCT2": -5
    }

    observations = {}
    traderData = ""

    state = TradingState(
        traderData,
        timestamp,
        listings,
        order_depths,
        own_trades,
        market_trades,
        position,
        observations
    )
    result = Trader().run(state)
    print(result)