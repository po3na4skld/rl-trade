"""

Portfolio object contains amount of money, list of opened orders, list of closed orders, current stocks
With Portfolio you can open and close orders (perform trading).

"""


class Portfolio:

    def __init__(self, bankroll):

        self.money_at_start = bankroll
        self.bankroll = bankroll

        self.opened_orders = []
        self.order_id = 0
        self.closed_orders = []

        self.coins = {}

    def open_order(self, currency, count, value):
        self.opened_orders.append({'currency': currency, 'count': count,
                                   'value': value, 'buy_price': count * value})

        self.bankroll -= count * value
        self.order_id += 1

        if currency in self.coins.keys():
            self.coins[currency]['count'] += count
        else:
            self.coins[currency] = {'count': count}

        return 0

    def close_order(self, sell_price):
        if not self.opened_orders:
            pass

        closed_order = self.opened_orders.pop(0)
        closed_order['sell_price'] = sell_price
        self.closed_orders.append(closed_order)

        self.bankroll += sell_price
        self.coins[closed_order['currency']]['count'] -= closed_order['count']
        return sell_price - closed_order['buy_price']

    def check_balance(self, value):
        return True if self.bankroll > value else False

    def get_profit(self):
        return self.money_at_start - self.bankroll

    def __str__(self):
        return f"Stocks: {self.coins}\nBankroll: {self.bankroll}\nCurrent profit {self.get_profit()}"
