# library imports
import os
import json
import numpy
import concurrent.futures
from random import choices, randint

# project imports
from agent import Agent
from pandemic import Pandemic
from population import Population, salary_values_weights, salary_values
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

        pandemic_accurance = []
        pandemic_accurance_weights = []
        # other option
        pandemic_accurance_mean = 10 # 33.33
        pandemic_accurance_std = 5 # 57.49

        pandemic_duration = []
        pandemic_duration_weights = []
        # other option
        pandemic_duration_mean = 3.28
        pandemic_duration_std = 3.12

        pandemic_death_percent = [0, 0.005, 0.01, 0.02, 0.03]
        pandemic_death_percent_weights = [0.7, 0.18, 0.07, 0.035, 0.015]

        # global parameters
        max_years = 100
        repeat_count = 10

        # generate multiple simulation and store in the end the mean and std of each result
        simulations = []

        for i in range(repeat_count):
            # generate population
            important_workers_count = 100
            non_important_workers_count = 900

            # the (0.9 + randint(0, 20)/100) is just to have a bit of change in the population
            agents = [Agent(working_type=Agent.IMPORTENT_WORKER,
                            salary=choices(salary_values, salary_values_weights, k=1)[0] * (0.9 + randint(0, 20)/100))
                      for i in range(important_workers_count)]
            agents.extend([Agent(working_type=Agent.NON_IMPORTENT_WORKER,
                                 salary=choices(salary_values, salary_values_weights, k=1)[0] * (0.9 + randint(0, 20)/100))
                           for i in range(non_important_workers_count)])

            # generate pandemic

            pandemics = []

            last_date = 0
            while last_date < max_years:
                #new_start_year = choices(pandemic_accurance, pandemic_accurance_weights, k=1)[0]
                #new_duration = choices(pandemic_duration, pandemic_duration_weights, k=1)[0]
                new_start_year = numpy.random.normal(numpy.mean(pandemic_accurance_mean), numpy.std(pandemic_accurance_std), 1)
                new_duration = numpy.random.normal(numpy.mean(pandemic_duration_mean), numpy.std(pandemic_duration_std), 1)
                new_kill_percent = choices(pandemic_death_percent, pandemic_death_percent_weights, k=1)[0]

                # fixes
                if new_start_year < 5:
                    new_start_year = 1 + randint(0, round(pandemic_accurance_mean))
                if new_duration < 1:
                    new_duration = 1

                new_pandemic = Pandemic(start_year=last_date + new_start_year,
                                        duration=new_duration,
                                        kill_percent=new_kill_percent)
                pandemics.append(new_pandemic)
                # update the history time
                last_date += new_start_year + new_duration

            # generate full simulation
            new_sim = Simulation(population=Population(agents=agents),
                                 pandemic_history=PandemicHistory(pandemics=pandemics),
                                 max_years=max_years,
                                 index=i+1)

            simulations.append(new_sim)

        # compute results for each simulation
        answer = []
        """
        # allow linear computing for debuging 
        for sim in simulations:
            answer.append(SimulatorRunner.perform_until_end_simulation(simulation=sim))
        """

        # run each one in process until they all finish
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

        mean_answer = sum(answer) / len(answer) if len(answer) > 0 else -1
        std_answer = numpy.std(answer) if len(answer) > 2 else -1

        # save results to the target file
        with open(target_path, "w") as target_file:
            json.dump({"simulation_duration": "{}".format(max_years),
                       "simulation_samples": "{}".format(repeat_count),
                       "crisis_years_mean:": "{:.3f}".format(mean_answer),
                       "crisis_years_std:": "{:.3f}".format(std_answer),
                       "crisis_years_data:": answer},
                      target_file,
                      indent=2)

    # Retrieve a single page and report the URL and contents
    @staticmethod
    def perform_until_end_simulation(simulation: Simulation):
        while not simulation.is_over():
            simulation.step()
        return simulation.years_pandemic_crisis


if __name__ == '__main__':
    SimulatorRunner.run(target_path=os.path.join(os.path.dirname(__file__), "run_answer.json"))
