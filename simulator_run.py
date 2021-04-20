# library imports
import os
import json
import concurrent.futures

# project imports
from agent import Agent
from pandemic import Pandemic
from population import Population
from pandemic_history import PandemicHistory
from simulator import Simulation


class SimulatorRunner:
    """
    Running multiple simulations by a given run settings file and save the results
    """

    MAX_THREADS = 4

    @staticmethod
    def run(target_path: str):
        """
        Run multiple simulations to answer all the configure requests
        :param target_path: path to the answer file to save the results of each simulation
        :return: saves results to the
        """
        # TODO: build population
        important_workers_count = 0
        agents = [Agent(working_type=Agent.IMPORTENT_WORKER,
                        salary=100) for i in range(important_workers_count)]
        agents = [Agent(working_type=Agent.IMPORTENT_WORKER,
                        salary=100) for i in range(important_workers_count)]
        # TODO: build pandemic history
        # TODO: build simulation
        simulations = []

        # run each one in process until they all finish
        answer = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=SimulatorRunner.MAX_THREADS) as executor:
            # Start the load operations and mark each future with its URL
            process = [executor.submit(SimulatorRunner.perform_until_end_simulation, sim) for sim in simulations]
            for future in concurrent.futures.as_completed(process):
                try:
                    data = future.result()
                except Exception as error:
                    print('Error occur, saying: {}'.format(error))
                else:
                    answer.append(data)

        # save results to the target file
        with open(target_path, "w") as target_file:
            json.dump(answer, target_file)

    # Retrieve a single page and report the URL and contents
    @staticmethod
    def perform_until_end_simulation(simulation: Simulation):
        while not simulation.is_over():
            simulation.step()
        return simulation.years_pandemic_crisis


if __name__ == '__main__':
    SimulatorRunner.run(config_path=os.path.join(os.path.dirname(__file__), "data", "multi_run_config.json"),
                        target_path=os.path.join(os.path.dirname(__file__), "answers", "multi_run_answer.json"))
