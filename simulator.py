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
                 max_years):
        self.population = population
        self.pandemic_history = pandemic_history
        self.time = 0

        self.funding = 0

        self.max_years = max_years

        # target parameters
        self.years_pandemic_crisis = 0

    @staticmethod
    def build_from_json(json_obj):
        answer = Simulation(population=Population.build_from_json(json_obj=json_obj["population"]),
                            pandemic_history=PandemicHistory.build_from_json(json_obj=json_obj["pandemic_history"]),
                            max_years=json_obj["max_years"])
        return answer

    def to_json(self):
        return {
            "population": self.population.to_json(),
            "pandemic_history": self.pandemic_history.to_json(),
            "max_years": self.max_years
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
            self.funding -= self.population.pandemic_pay()
            self.population.kill(kill_percent=pandemic.kill_percent)
            if self.funding <= 0:
                # update the target parameters
                self.years_pandemic_crisis += 1
        else:
            for agent in self.population:
                self.funding += agent.pay()

        # count this time
        self.time += 1

        # just for debug
        print("Simulation at {} years ".format(self.time))

    def is_over(self):
        return self.time >= self.max_years

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<Simulation | time = {}>".format(self.time)