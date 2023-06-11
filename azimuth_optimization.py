import pandas as pd

from bayes_opt import BayesianOptimization
from bayes_opt import SequentialDomainReductionTransformer

NREL_API_KEY = "YOUR API KEY HERE"

INITIAL_POINTS = 10
NUMBER_OF_ITERATIONS = 20
PVWATTS_API_URL = "https://developer.nrel.gov/api/pvwatts/v8.json?api_key={api}&azimuth={azimuth}&system_capacity=1000&losses=14&array_type=0&module_type=0&&&tilt=45&address=copenhagen,dk&&albedo=0.3&&timeframe=hourly&&dataset=intl"
SPOT_PRICE_DATA = "spot_prices_dk2/2022_prices.csv"


class Optimizer:
    def profit_target_function(self, azimuth):
        url = PVWATTS_API_URL.format(
            api=NREL_API_KEY, azimuth=azimuth)

        production = pd.read_json(url, lines=True)
        prices = pd.read_csv(
            SPOT_PRICE_DATA).sort_values('f0_')
        prices_and_production = pd.merge(pd.DataFrame(production.head()['outputs'][0]['ac'], columns=[
            'production']), prices, left_index=True, right_index=True)
        return sum(prices_and_production['price_eur'] * prices_and_production['production'])

    def opt_PC(self):
        pbounds = {'azimuth': (90, 270)}

        optimizer = BayesianOptimization(
            f=self.profit_target_function, pbounds=pbounds, verbose=2, random_state=1, allow_duplicate_points=True, bounds_transformer=SequentialDomainReductionTransformer())
        optimizer.set_gp_params(alpha=1e-3, n_restarts_optimizer=20)
        optimizer.maximize(INITIAL_POINTS, NUMBER_OF_ITERATIONS)
        return optimizer.max

    def output(self):
        return self.opt_PC()


if __name__ == "__main__":
    optimizer = Optimizer()
    print(optimizer.output())
