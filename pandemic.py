# library imports

# project imports


class Pandemic:
    """
    Single Pandemic event
    """

    def __init__(self,
                 start_year,
                 duration,
                 kill_percent):
        # state
        self.start_year = start_year
        self.duration = duration
        self.kill_percent = kill_percent

    @staticmethod
    def build_from_json(json_obj):
        """
        build instance of this process from json file
        """
        return Pandemic(start_year=json_obj["start_year"],
                        duration=json_obj["duration"],
                        kill_percent=json_obj["kill_percent"])

    def to_json(self):
        return {
            "start_year": self.start_year,
            "duration": self.duration,
            "kill_percent": self.kill_percent
        }

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<Pandemic: ({}, {:.2f}%)>".format(self.duration, self.kill_percent * 100)

    # getters #

    # end - getters #

    # state change #

    # end - state change #
