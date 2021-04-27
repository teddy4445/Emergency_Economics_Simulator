import math

from pandemic_history import PandemicHistory
from population import Population


class Simulation:
    """
    A single simulation of the dynamics given everything
    """

    def __init__(self,
                 population,
                 pandemic_history,
                 max_years,
                 positive_money_rate: float,
                 negative_money_rate: float,
                 tax_rate: float = 0.025,
                 payment_policy: float = 0.75,
                 index: int = 0,
                 debug: bool = False):
        self.index = index
        self.population = population
        self.pandemic_history = pandemic_history
        self.time = 0

        self.funding = 0

        self.max_years = max_years

        # target parameters
        self.years_pandemic_crisis = 0

        # properties
        self.payment_policy = payment_policy
        self.tax_rate = tax_rate
        self.positive_money_rate = positive_money_rate
        self.negative_money_rate = negative_money_rate

        self.debug = debug
        self.history = []

    @staticmethod
    def build_from_json(json_obj):
        answer = Simulation(population=Population.build_from_json(json_obj=json_obj["population"]),
                            pandemic_history=PandemicHistory.build_from_json(json_obj=json_obj["pandemic_history"]),
                            max_years=json_obj["max_years"],
                            tax_rate=json_obj["tax_rate"],
                            payment_policy=json_obj["payment_policy"],
                            positive_money_rate=json_obj["positive_money_rate"],
                            negative_money_rate=json_obj["negative_money_rate"])
        return answer

    def to_json(self):
        return {
            "population": self.population.to_json(),
            "pandemic_history": self.pandemic_history.to_json(),
            "max_years": self.max_years,
            "tax_rate": self.tax_rate,
            "payment_policy": self.payment_policy,
            "positive_money_rate": self.positive_money_rate,
            "negative_money_rate": self.negative_money_rate
        }

    def clear(self):
        """
        make sure the simulator is empty for next run
        """
        self.population = None
        self.pandemic_history = None
        self.max_years = 0
        self.time = 0

    def run(self):
        while not self.is_over():
            self.step()
        return self.years_pandemic_crisis

    def step(self):
        """
        perform the full step of population interactions and pandemic
        """

        if self.time > self.max_years:
            return
        pandemic = self.pandemic_history.in_pandemic(time=self.time)
        if pandemic is not None:
            self.funding -= self.population.pandemic_pay(tax_rate=self.tax_rate,
                                                         payment_policy=self.payment_policy)
            self.population.kill(kill_percent=pandemic.kill_percent / pandemic.duration)
        else:
            for agent in self.population:
                self.funding += agent.pay(self.tax_rate)

        # if crisis year
        if self.funding <= 0:
            # update the target parameters
            self.years_pandemic_crisis += 1
            self.funding *= (1 + self.positive_money_rate)
        else:
            self.funding *= (1 + self.negative_money_rate)

        # count this time
        self.time += 1
        self.history.append([self.time, self.years_pandemic_crisis, self.funding])

        # grow population
        self.population.burn()

        # just for debug
        print("Simulation #{} at {} years (funding: {:.2f}, crisis: {})".format(self.index, self.time, self.funding, self.years_pandemic_crisis))

    def is_over(self):
        return self.time >= self.max_years

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<Simulation | time = {}>".format(self.time)
