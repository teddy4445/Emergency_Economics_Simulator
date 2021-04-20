# library imports

# project imports


class Agent:
    """
    Single Agent in the population
    """

    IMPORTENT_WORKER = 1
    NON_IMPORTENT_WORKER = 2

    def __init__(self,
                 working_type,
                 salary):
        # state
        self.working_type = working_type
        self.salary = salary

    @staticmethod
    def build_from_json(json_obj):
        """
        build instance of this process from json file
        """
        return Agent(working_type=json_obj["working_type"],
                     salary=json_obj["salary"])

    def to_json(self):
        return {
            "working_type": self.working_type,
            "salary": self.salary
        }

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<Agent: working type - {}>".format("Important" if Agent.IMPORTENT_WORKER == self.working_type else "Unimportant")

    # getters #

    def pay(self):
        return 0.1 * self.salary

    # end - getters #

    # state change #

    # end - state change #
