# library imports
import os
import json
import numpy
import matplotlib
import concurrent.futures
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt
from random import choices, randint
from sklearn.metrics import r2_score
from scipy.interpolate import interp2d
from sklearn.linear_model import LinearRegression

# project imports
from agent import Agent
from pandemic import Pandemic
from simulator import Simulation
from pandemic_history import PandemicHistory
from population import Population, salary_values_weights, salary_values, ages_values, ages_values_weights


class SimulatorRunner:
    """
    Running multiple simulations by a given run settings file and save the results
    """

    MAX_THREADS = 4

    @staticmethod
    def run_sensitivity(target_path_csv: str,
                        target_path_png: str):

        text_to_write = "tax_rate,helping_policy,crisis_years_mean,crisis_years_std,funding_mean,funding_std,outcome_mean,outcome_std\n"
        answer_data = []

        # global parameters
        max_years = 100
        repeat_count = 5

        tax_rates = []
        payment_policies = []
        global_crisis = []
        global_funding = []
        global_outcome = []

        for tax_rate in [0.0025 * i for i in range(20)]:  # 8
            for payment_policy in [0.75 + 0.025 * i for i in range(11)]:  # 6
                print("Start tax rate = {}, payment_policy = {}".format(tax_rate, payment_policy))
                # prepare simulations
                simulations = SimulatorRunner.prepare_simulations(max_years=max_years,
                                                                  repeat_count=repeat_count,
                                                                  tax_rate=2*tax_rate,
                                                                  payment_policy=payment_policy)

                # compute results for each simulation
                crisis = []
                outcomes = []
                funding = []
                # run each one in process until they all finish
                with concurrent.futures.ThreadPoolExecutor(max_workers=SimulatorRunner.MAX_THREADS) as executor:
                    # Start the load operations and mark each future with its URL
                    process = [executor.submit(SimulatorRunner.perform_until_end_simulation, sim) for sim in simulations]
                    for future in concurrent.futures.as_completed(process):
                        try:
                            crisis_years, outcome, history = future.result()
                        except Exception as error:
                            print('Error occur, saying: {}'.format(error))
                        else:
                            crisis.append(crisis_years)
                            outcomes.append(outcome)
                            funding.append(history[-1][2])
                            tax_rates.append(tax_rate)
                            payment_policies.append(payment_policy)

                mean_crisis = sum(crisis) / len(crisis) if len(crisis) > 0 else -1
                std_crisis = numpy.std(crisis) if len(crisis) > 2 else -1
                mean_funding = sum(funding) / len(funding) if len(funding) > 0 else -1
                std_funding = numpy.std(funding) if len(funding) > 2 else -1
                mean_outcome = sum(outcomes) / len(outcomes) if len(outcomes) > 0 else -1
                std_outcome = numpy.std(outcomes) if len(outcomes) > 2 else -1

                global_crisis.append(mean_crisis)
                global_funding.append(mean_funding)
                global_outcome.append(mean_outcome)

                text_to_write += "{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n".format(tax_rate, payment_policy, mean_crisis, std_crisis, mean_funding, std_funding, mean_outcome, std_outcome)
                answer_data.append([tax_rate, payment_policy, mean_crisis, std_crisis, mean_funding, std_funding, mean_outcome, std_outcome])
                print("Finish loop for tax_rate = {:.3f} and payment_policy = {:.3f}\n\n".format(tax_rate, payment_policy))

        # save results to the target file
        with open(target_path_csv, "w") as target_file:
            target_file.write(text_to_write)

        print("tax: {}, payment: {}, crisis: {}, outcome: {}"
              .format(len(tax_rates),
                      len(payment_policies),
                      len(global_crisis),
                      len(global_outcome)))

        x = numpy.unique(tax_rates)
        y = numpy.unique(payment_policies)
        X, Y = numpy.meshgrid(x, y)
        Z = numpy.asarray(global_crisis).reshape(len(y), len(x))
        pc = plt.pcolormesh(X, numpy.flip(Y), Z, vmin=0, vmax=numpy.max(Z))
        plt.xlabel('C [1]', fontsize=16)
        plt.ylabel('R [1]', fontsize=16)
        plt.colorbar(pc)
        plt.savefig(target_path_png.replace(".png", "_crisis_year.png"))
        plt.close()

        x = numpy.unique(tax_rates)
        y = numpy.unique(payment_policies)
        X, Y = numpy.meshgrid(x, y)
        Z = numpy.asarray(global_funding).reshape(len(y), len(x))
        pc = plt.pcolormesh(X, numpy.flip(Y), Z, vmin=numpy.min(Z), vmax=numpy.max(Z))
        plt.xlabel('C [1]', fontsize=16)
        plt.ylabel('R [1]', fontsize=16)
        plt.colorbar(pc)
        plt.savefig(target_path_png.replace(".png", "_funding.png"))
        plt.close()

        x = numpy.unique(tax_rates)
        y = numpy.unique(payment_policies)
        X, Y = numpy.meshgrid(x, y)
        Z = numpy.asarray(global_outcome).reshape(len(y), len(x))
        pc = plt.pcolormesh(X, numpy.flip(Y), Z, vmin=numpy.min(Z), vmax=numpy.max(Z))
        plt.xlabel('C [1]', fontsize=16)
        plt.ylabel('R [1]', fontsize=16)
        plt.colorbar(pc)
        plt.savefig(target_path_png.replace(".png", "_outcome.png"))
        plt.close()

    @staticmethod
    def run(target_path: str):
        """
        Run multiple simulations to answer all the configure requests
        :param target_path: path to the answer file to save the results of each simulation
        :return: saves results to the
        """
        # global parameters
        max_years = 100
        repeat_count = 50
        # prepare simulations
        simulations = SimulatorRunner.prepare_simulations(max_years=max_years,
                                                          repeat_count=repeat_count,
                                                          tax_rate=0.1,
                                                          payment_policy=0.8)

        # compute results for each simulation
        answer = []
        outcomes = []
        histories = []
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
                    crisis_years, outcome, history = future.result()
                except Exception as error:
                    print('Error occur, saying: {}'.format(error))
                else:
                    answer.append(crisis_years)
                    outcomes.append(outcome)
                    histories.append(history)

        mean_answer = sum(answer) / len(answer) if len(answer) > 0 else -1
        std_answer = numpy.std(answer) if len(answer) > 2 else -1

        mean_outcome = sum(outcomes) / len(outcomes) if len(outcomes) > 0 else -1
        std_outcome = numpy.std(outcomes) if len(outcomes) > 2 else -1

        # save results to the target file
        with open(target_path, "w") as target_file:
            json.dump({"simulation_duration": "{}".format(max_years),
                       "simulation_samples": "{}".format(repeat_count),
                       "crisis_years_mean:": "{:.3f}".format(mean_answer),
                       "crisis_years_std:": "{:.3f}".format(std_answer),
                       "out_mean:": "{:.3f}".format(mean_outcome),
                       "out_std:": "{:.3f}".format(std_outcome),
                       "crisis_years_data:": answer},
                      target_file,
                      indent=2)

        ground_answer_time = [val[0] for val in histories[0]]
        histories_transpose = [[histories[j][i] for j in range(len(histories))] for i in range(len(histories[0]))]
        ground_answer_crisis_years_mean = [numpy.mean([item[1] for item in time_obj]) for time_obj in histories_transpose]
        ground_answer_crisis_years_std = [numpy.std([item[1] for item in time_obj]) for time_obj in histories_transpose]
        ground_answer_funding_years_mean = [numpy.mean([item[2] for item in time_obj]) for time_obj in histories_transpose]
        ground_answer_funding_years_std = [numpy.std([item[2] for item in time_obj]) for time_obj in histories_transpose]
        ground_answer_outcome_years_mean = [numpy.mean([item[3] for item in time_obj]) for time_obj in histories_transpose]
        ground_answer_outcome_years_std = [numpy.std([item[3] for item in time_obj]) for time_obj in histories_transpose]

        # single plots

        fig, ax1 = plt.subplots()
        ax1.plot(ground_answer_time,
                 ground_answer_crisis_years_mean,
                 "-o",
                 markersize=2,
                 color="r")
        ax1.fill_between(numpy.asarray(ground_answer_time),
                         numpy.asarray(ground_answer_crisis_years_mean) + numpy.asarray(ground_answer_crisis_years_std),
                         numpy.asarray(ground_answer_crisis_years_mean) - numpy.asarray(ground_answer_crisis_years_std),
                         alpha=0.5,
                         color="r")
        ax1.tick_params(axis='y')
        plt.xlabel('Years [t]', fontsize=16)
        plt.ylabel('Crisis years [1]', fontsize=16)
        plt.ylim((0, 70))
        plt.xlim((0, 100))
        fig.tight_layout()
        ax1.grid()
        ax1.yaxis.set_ticks_position('left')
        ax1.xaxis.set_ticks_position('bottom')
        fig.savefig(target_path.replace(".json", "_crisis_run.png"))
        plt.close()


        fig, ax2 = plt.subplots()
        ax2.plot(ground_answer_time,
                 ground_answer_funding_years_mean,
                 "-o",
                 markersize=2,
                 color="b")
        ax2.fill_between(numpy.asarray(ground_answer_time),
                         numpy.asarray(ground_answer_funding_years_mean) + numpy.asarray(ground_answer_funding_years_std),
                         numpy.asarray(ground_answer_funding_years_mean) - numpy.asarray(ground_answer_funding_years_std),
                         alpha=0.5,
                         color="b")
        ax2.tick_params(axis='y')
        plt.xlabel('Years [t]', fontsize=16)
        plt.ylabel('Funding in Israeli Shekels [1]', fontsize=16)
        plt.ylim((-100000000, 100000000))
        plt.xlim((0, 100))
        ax2.grid()
        fig.tight_layout()
        ax2.yaxis.set_ticks_position('left')
        ax2.xaxis.set_ticks_position('bottom')
        fig.savefig(target_path.replace(".json", "_funding_run.png"))
        plt.close()


        fig, ax3 = plt.subplots()
        ax3.plot(ground_answer_time,
                 ground_answer_outcome_years_mean,
                 "-o",
                 markersize=2,
                 color="g")
        ax3.fill_between(numpy.asarray(ground_answer_time),
                         numpy.asarray(ground_answer_outcome_years_mean) + 2 * numpy.asarray(ground_answer_outcome_years_std),
                         numpy.asarray(ground_answer_outcome_years_mean) - 2 * numpy.asarray(ground_answer_outcome_years_std),
                         alpha=0.5,
                         color="g")
        ax3.tick_params(axis='y')
        plt.xlabel('Years [t]', fontsize=16)
        plt.ylabel('Outcome in Israeli Shekels [1]', fontsize=16)
        plt.ylim((0, 10000000000))
        plt.xlim((0, 100))
        ax3.grid()
        fig.tight_layout()
        ax3.yaxis.set_ticks_position('left')
        ax3.xaxis.set_ticks_position('bottom')
        fig.savefig(target_path.replace(".json", "_outcome_run.png"))
        plt.close()

    @staticmethod
    def prepare_simulations(max_years,
                            repeat_count,
                            tax_rate,
                            payment_policy=0.75):
        pandemic_accurance = []
        pandemic_accurance_weights = []
        # other option
        pandemic_accurance_mean = 17.33 # 33.33
        pandemic_accurance_std = 6.312 # 57.49

        pandemic_duration = []
        pandemic_duration_weights = []
        # other option
        pandemic_duration_mean = 3.28
        pandemic_duration_std = 3.12

        pandemic_death_percent = [0, 0.005, 0.01, 0.02, 0.03]
        pandemic_death_percent_weights = [0.7, 0.18, 0.07, 0.035, 0.015]

        # generate multiple simulation and store in the end the mean and std of each result
        simulations = []

        for i in range(repeat_count):
            # generate population
            important_workers_count = 200
            non_important_workers_count = 800

            # the (0.9 + randint(0, 20)/100) is just to have a bit of change in the population
            agents = [Agent(working_type=Agent.IMPORTENT_WORKER,
                            age=round(choices(ages_values, ages_values_weights, k=1)[0] * (0.9 + randint(0, 20)/100)),
                            salary=round(choices(salary_values, salary_values_weights, k=1)[0] * (0.9 + randint(0, 20)/100)))
                      for i in range(important_workers_count)]
            agents.extend([Agent(working_type=Agent.NON_IMPORTENT_WORKER,
                                 age=round(choices(ages_values, ages_values_weights, k=1)[0] * (0.9 + randint(0, 20)/100)),
                                 salary=round(choices(salary_values, salary_values_weights, k=1)[0] * (0.9 + randint(0, 20)/100)))
                           for i in range(non_important_workers_count)])

            # generate pandemic

            pandemics = []

            last_date = 0
            while last_date < max_years:
                #new_start_year = choices(pandemic_accurance, pandemic_accurance_weights, k=1)[0]
                #new_duration = choices(pandemic_duration, pandemic_duration_weights, k=1)[0]
                ###new_start_year = round(numpy.random.normal(pandemic_accurance_mean, pandemic_accurance_std, 1)[0])
                ###new_duration = round(numpy.random.normal(pandemic_duration_mean, pandemic_duration_std, 1)[0])
                ###new_kill_percent = choices(pandemic_death_percent, pandemic_death_percent_weights, k=1)[0]
                new_start_year = round(numpy.random.normal(pandemic_accurance_mean, 0, 1)[0])
                new_duration = round(numpy.random.normal(pandemic_duration_mean, 0, 1)[0])
                new_kill_percent = 0.02

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

            positive_money_rate = 0.015
            negative_money_rate = 0.03

            # generate full simulation
            new_sim = Simulation(population=Population(agents=agents),
                                 pandemic_history=PandemicHistory(pandemics=pandemics),
                                 max_years=max_years,
                                 index=i+1,
                                 payment_policy=payment_policy,
                                 tax_rate=tax_rate,
                                 positive_money_rate=positive_money_rate,
                                 negative_money_rate=negative_money_rate,
                                 debug=True)

            simulations.append(new_sim)
        return simulations

    # Retrieve a single page and report the URL and contents
    @staticmethod
    def perform_until_end_simulation(simulation: Simulation):
        while not simulation.is_over():
            simulation.step()
        return simulation.years_pandemic_crisis, simulation.outcome, simulation.history


if __name__ == '__main__':
    #SimulatorRunner.run(target_path=os.path.join(os.path.dirname(__file__), "results", "run_answer.json"))
    SimulatorRunner.run_sensitivity(target_path_csv=os.path.join(os.path.dirname(__file__), "results", "run_sensitivity_tax_rate.csv"),       target_path_png=os.path.join(os.path.dirname(__file__), "results", "run_sensitivity_tax_rate.png"))
