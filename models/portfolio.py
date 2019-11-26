

class Portfolio:

    def __init__(self, bankroll):

        self.money_at_start = bankroll
        self.bankroll = bankroll

        self.opened_orders = {}
        self.order_id = 0
        self.closed_orders = []

        self.portfolio = {}

    def open_order(self, currency, count, value, buy_price):
        self.opened_orders[self.order_id] = {'currency': currency, 'count': count,
                                             'value': value, 'buy_price': buy_price}

        self.bankroll -= buy_price
        self.order_id += 1

        if currency in self.portfolio.keys():
            self.portfolio[currency]['count'] += count
        else:
            self.portfolio[currency] = {'count': count}

    def close_order(self, order_id, sell_price):
        if order_id not in self.opened_orders.keys():
            raise KeyError(f"Order with {order_id} does not exist.")

        closed_order = self.opened_orders.pop(order_id)
        closed_order['sell_price'] = sell_price
        self.closed_orders.append(closed_order)

        self.bankroll += sell_price
        self.portfolio[closed_order['currency']]['count'] -= closed_order['count']

    def check_balance(self, value):
        return True if self.bankroll > value else False

    def get_profit(self):
        return self.money_at_start - self.bankroll

    def __str__(self):
        return f"Portfolio: {self.portfolio}\nBankroll: {self.bankroll}"
