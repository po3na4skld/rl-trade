
class Environment:

    def buy(self):
        raise NotImplementedError

    def sell(self):
        raise NotImplementedError

    def hold(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def update_state(self):
        raise NotImplementedError
