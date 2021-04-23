# library imports

# project imports


CRISIS_EVENT_TAX_RATE = 0.01


class Agent:
    """
    Single Agent in the population
    """

    IMPORTENT_WORKER = 1
    NON_IMPORTENT_WORKER = 2

    def __init__(self,
                 age,
                 working_type,
                 salary):
        # state
        self.age = age
        self.working_type = working_type
        self.salary = salary

    @staticmethod
    def build_from_json(json_obj):
        """
        build instance of this process from json file
        """
        return Agent(working_type=json_obj["working_type"],
                     salary=json_obj["salary"],
                     age=json_obj["age"])

    def to_json(self):
        return {
            "working_type": self.working_type,
            "salary": self.salary,
            "age": self.age
        }

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<Agent: working type - {}>".format("Important" if Agent.IMPORTENT_WORKER == self.working_type else "Unimportant")

    # getters #

    def pay(self,
            tax_rate: float):
        # https://www.kolzchut.org.il/he/%D7%9E%D7%93%D7%A8%D7%92%D7%95%D7%AA_%D7%9E%D7%A1_%D7%94%D7%9B%D7%A0%D7%A1%D7%94
        answer = 0
        if 0 <= self.salary < 6290:
            answer = 0.1 * self.salary
        elif 6290 <= self.salary < 9030:
            answer = 0.1 * 6290 + (self.salary - 6290) * 0.14
        elif 9030 <= self.salary < 14490:
            answer = 0.1 * 6290 + 0.14 * (9030 - 6290) + (self.salary - 9030) * 0.20
        elif 14490 <= self.salary < 20140:
            answer = 0.1 * 6290 + 0.14 * (9030 - 6290) + 0.20 * (14490 - 9030) + (self.salary - 14490) * 0.31
        elif 20140 <= self.salary < 41910:
            answer = 0.1 * 6290 + 0.14 * (9030 - 6290) + 0.20 * (14490 - 9030) + 0.31 * (20140 - 14490) + (self.salary - 20140) * 0.35
        elif 41910 <= self.salary < 53970:
            answer = 0.1 * 6290 + 0.14 * (9030 - 6290) + 0.20 * (14490 - 9030) + 0.31 * (20140 - 14490) + 0.35 * (41910 - 20140) + (self.salary - 41910) * 0.47
        else:  # elif 53970 <= self.salary:
            answer = 0.1 * 6290 + 0.14 * (9030 - 6290) + 0.20 * (14490 - 9030) + 0.31 * (20140 - 14490) + 0.35 * (41910 - 20140) + 0.47 * (53970 - 41910) + (self.salary - 53970) * 0.5

        return answer * tax_rate

    # end - getters #

    # state change #

    # end - state change #
