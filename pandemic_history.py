# library imports
from pandemic import Pandemic
# project imports


class PandemicHistory:
    """
    Single Pandemic event
    """

    def __init__(self,
                 pandemics):
        # state
        self.pandemics = pandemics
        sorted(self.pandemics, key=lambda x: x.start_year)

    @staticmethod
    def build_from_json(json_obj):
        """
        build instance of this process from json file
        """
        return PandemicHistory(pandemics=json_obj["pandemics"])

    def to_json(self):
        return {
            "pandemics": self.pandemics
        }

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<PandemicHistory -> |P| = {}>".format(len(self.pandemics))

    def __iter__(self):
        return self

    def __reversed__(self):
        return self.pandemics[::-1]

    def __next__(self):
        self._iter_index += 1
        if self._iter_index >= len(self.pandemics):
            self._iter_index = -1
            raise StopIteration
        return self.pandemics[self._iter_index]

    def size(self):
        return len(self.pandemics)

    def clear(self):
        self.pandemics.clear()

    # getters #

    def in_pandemic(self, time):
        for pandemic in self.pandemics:
            if pandemic.start_year <= time <= pandemic.start_year + pandemic.duration:
                return pandemic
        return None

    # end - getters #

    # state change #

    # end - state change #
