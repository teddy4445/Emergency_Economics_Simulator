import math
import random

from agent import Agent


BURN_RATE = 0.02
IMPORTANT_EMPLOY_RATE = 0.5


class Population:
    """
    Basically, a static list (may be changed later) of agents.
    This class adds common methods on the entire population.
    """

    def __init__(self,
                 agents: list = None):
        self.agents = agents if agents is not None else []

        # technical var
        self._iter_index = -1

    @staticmethod
    def build_from_json(json_obj: list):
        agents = []
        [agents.extend([Agent.build_from_json(meta_agent["agent"])
                        for i in range(meta_agent["count"])])
         for meta_agent in json_obj]
        return Population(agents=agents)

    def to_json(self):
        return [{
            "agent": agent.to_json(),
            "count": 1
        }
            for agent in self.agents]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<Population -> |P| = {}>".format(len(self.agents))

    def __iter__(self):
        return self

    def __reversed__(self):
        return self.agents[::-1]

    def __next__(self):
        self._iter_index += 1
        if self._iter_index >= len(self.agents):
            self._iter_index = -1
            raise StopIteration
        return self.agents[self._iter_index]

    def size(self):
        return len(self.agents)

    def clear(self):
        self.agents.clear()

    # logical functions #

    def burn(self):
        self.agents.extend([Agent(working_type=Agent.IMPORTENT_WORKER, salary=100) for i in range(math.ceil(len(self.agents)*BURN_RATE*IMPORTANT_EMPLOY_RATE))])
        self.agents.extend([Agent(working_type=Agent.NON_IMPORTENT_WORKER, salary=100) for i in range(math.ceil(len(self.agents)*BURN_RATE*(1-IMPORTANT_EMPLOY_RATE)))])

    def kill(self,
             kill_percent: float):
        self.agents = random.sample(self.agents, int(len(self.agents) * (1 - kill_percent)))

    def pandemic_pay(self):
        payment = 0
        for agent in self.agents:
            if agent.working_type == Agent.IMPORTENT_WORKER:
                payment += 100
            else:
                payment += 200
        return payment

    def distribution(self,
                     key_function):
        """
        :return: the distribution of the population's according to some property of an agent
        """
        answer = {}
        for agent in self.agents:
            if key_function == "get_epidemic_state":
                key = agent.get_epidemic_state()
            elif key_function == "get_location":
                key = agent.get_location()
            elif key_function == "get_state_vector_str":
                key = agent.get_state_vector_str()
            else:
                key = ""
            try:
                answer[key] += 1
            except:
                answer[key] = 1
        return answer

    # end - logical functions #
