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

        text_to_write = "tax_rate,helping_policy,crisis_years_mean,crisis_years_std,funding_mean,funding_std\n"
        answer_data = []

        # global parameters
        max_years = 100
        repeat_count = 15

        tax_rates = []
        payment_policies = []
        global_crisis = []
        global_funding = []

        for tax_rate in [0.0025 * i for i in range(20)]:  # 8
            for payment_policy in [0.75 + 0.025 * i for i in range(11)]:  # 6
                print("Start tax rate = {}, payment_policy = {}".format(tax_rate, payment_policy))
                # prepare simulations
                simulations = SimulatorRunner.prepare_simulations(max_years=max_years,
                                                                  repeat_count=repeat_count,
                                                                  tax_rate=tax_rate,
                                                                  payment_policy=payment_policy)

                # compute results for each simulation
                crisis = []
                funding = []
                # run each one in process until they all finish
                with concurrent.futures.ThreadPoolExecutor(max_workers=SimulatorRunner.MAX_THREADS) as executor:
                    # Start the load operations and mark each future with its URL
                    process = [executor.submit(SimulatorRunner.perform_until_end_simulation, sim) for sim in simulations]
                    for future in concurrent.futures.as_completed(process):
                        try:
                            crisis_years, history = future.result()
                        except Exception as error:
                            print('Error occur, saying: {}'.format(error))
                        else:
                            crisis.append(crisis_years)
                            funding.append(history[-1][2])
                            tax_rates.append(tax_rate)
                            payment_policies.append(payment_policy)

                mean_crisis = sum(crisis) / len(crisis) if len(crisis) > 0 else -1
                std_crisis = numpy.std(crisis) if len(crisis) > 2 else -1
                mean_funding = sum(funding) / len(funding) if len(funding) > 0 else -1
                std_funding = numpy.std(funding) if len(funding) > 2 else -1

                global_crisis.append(mean_crisis)
                global_funding.append(mean_funding)

                text_to_write += "{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n".format(tax_rate, payment_policy, mean_crisis, std_crisis, mean_funding, std_funding)
                answer_data.append([tax_rate, payment_policy, mean_crisis, std_crisis, mean_funding, std_funding])
                print("Finish loop for tax_rate = {:.3f} and payment_policy = {:.3f}\n\n".format(tax_rate, payment_policy))

        # save results to the target file
        with open(target_path_csv, "w") as target_file:
            target_file.write(text_to_write)

        print("tax: {}, payment: {}, crisis: {}"
              .format(len(tax_rates),
                      len(payment_policies),
                      len(global_crisis)))

        x = numpy.unique(tax_rates)
        y = numpy.unique(payment_policies)
        X, Y = numpy.meshgrid(x, y)
        Z = numpy.asarray(global_crisis).reshape(len(y), len(x))
        pc = plt.pcolormesh(X, Y, Z, vmin=0, vmax=numpy.max(Z))
        plt.colorbar(pc)
        plt.savefig(target_path_png.replace(".png", "_crisis_year.png"))
        plt.close()

        x = numpy.unique(tax_rates)
        y = numpy.unique(payment_policies)
        X, Y = numpy.meshgrid(x, y)
        Z = numpy.asarray(global_funding).reshape(len(y), len(x))
        pc = plt.pcolormesh(X, Y, Z, vmin=numpy.min(Z), vmax=numpy.max(Z))
        plt.colorbar(pc)
        plt.savefig(target_path_png.replace(".png", "_funding.png"))
        plt.close()

        """
        reg = LinearRegression().fit(numpy.asarray([[answer_data[i][0]] for i in range(len(answer_data))]),
                                     numpy.asarray([answer_data[i][1] for i in range(len(answer_data))]))

        # save plot
        fig, ax = plt.subplots()
        ax.errorbar([answer_data[i][0] for i in range(len(answer_data))],
                    [answer_data[i][1] for i in range(len(answer_data))],
                    yerr=[answer_data[i][2] for i in range(len(answer_data))],
                    marker="o",
                    markersize=4,
                    color="k")
        ax.plot([min([answer_data[i][0] for i in range(len(answer_data))]),
                 max([answer_data[i][0] for i in range(len(answer_data))])],
                [min([answer_data[i][1] for i in range(len(answer_data))]),
                 max([answer_data[i][1] for i in range(len(answer_data))])],
                color="b",
                label="$R^2 = {:.2f}$ | $y = {:.2f} x + {:.2f}$".format(r2_score(numpy.asarray([answer_data[i][1] for i in range(len(answer_data))]),
                                                                                 reg.predict(numpy.asarray([[answer_data[i][0]] for i in range(len(answer_data))]))),
                                                                        reg.coef_[0],
                                                                        reg.intercept_))
        ax.set(xlabel='Tax rate [1]',
               ylabel='Crisis years [1]',
               title='Tax rate analysis over crisis years')
        ax.legend()
        ax.set_ylim([0, max_years])
        ax.grid()
        fig.savefig(target_path_png)
        plt.close()
        """

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
                                                          tax_rate=0.07,
                                                          payment_policy=0.75)

        # compute results for each simulation
        answer = []
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
                    crisis_years, history = future.result()
                except Exception as error:
                    print('Error occur, saying: {}'.format(error))
                else:
                    answer.append(crisis_years)
                    histories.append(history)

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

        ground_answer_time = [val[0] for val in histories[0]]
        histories_transpose = [[histories[j][i] for j in range(len(histories))] for i in range(len(histories[0]))]
        ground_answer_crisis_years_mean = [numpy.mean([item[1] for item in time_obj]) for time_obj in histories_transpose]
        ground_answer_crisis_years_std = [numpy.std([item[1] for item in time_obj]) for time_obj in histories_transpose]
        ground_answer_funding_years_mean = [numpy.mean([item[2] for item in time_obj]) for time_obj in histories_transpose]
        ground_answer_funding_years_std = [numpy.std([item[2] for item in time_obj]) for time_obj in histories_transpose]

        # save plot
        fig, ax1 = plt.subplots()
        ax1.errorbar(ground_answer_time,
                    ground_answer_crisis_years_mean,
                    yerr=ground_answer_crisis_years_std,
                    marker="o",
                    markersize=2,
                    color="r")
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax1.set(xlabel='Tax rate [1]',
                ylabel='Crisis years [1]')
        ax1.grid()
        ax2 = ax1.twinx()
        ax2.errorbar(ground_answer_time,
                    ground_answer_funding_years_mean,
                    yerr=ground_answer_funding_years_std,
                    marker="x",
                    markersize=2,
                    color="b")
        ax2.tick_params(axis='y', labelcolor="tab:blue")
        ax2.set(ylabel='Funding in Israeli Shekels [1]')
        fig.tight_layout()
        fig.savefig(target_path.replace("json", "png"))
        plt.close()

        # single plots

        fig, ax1 = plt.subplots()
        ax1.errorbar(ground_answer_time,
                    ground_answer_crisis_years_mean,
                    yerr=ground_answer_crisis_years_std,
                    marker="o",
                    markersize=3,
                    color="r")
        ax1.tick_params(axis='y')
        ax1.set(xlabel='Tax rate [1]',
                ylabel='Crisis years [1]')
        plt.ylim((0, 100))
        ax1.grid()
        fig.tight_layout()
        fig.savefig(target_path.replace(".json", "_crisis_run.png"))
        plt.close()


        fig, ax2 = plt.subplots()
        ax2.errorbar(ground_answer_time,
                    ground_answer_funding_years_mean,
                    yerr=ground_answer_funding_years_std,
                    marker="x",
                    markersize=3,
                    color="b")
        ax2.tick_params(axis='y')
        ax2.set(xlabel='Tax rate [1]',
                ylabel='Funding in Israeli Shekels [1]')
        ax1.grid()
        plt.ylim((-100000000, 100000000))
        fig.tight_layout()
        fig.savefig(target_path.replace(".json", "_funding_run.png"))
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
                new_start_year = round(numpy.random.normal(pandemic_accurance_mean, pandemic_accurance_std, 1)[0])
                new_duration = round(numpy.random.normal(pandemic_duration_mean, pandemic_duration_std, 1)[0])
                ###new_start_year = round(numpy.random.normal(pandemic_accurance_mean, 0, 1)[0])
                ###new_duration = round(numpy.random.normal(pandemic_duration_mean, 0, 1)[0])
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
        return simulation.years_pandemic_crisis, simulation.history


if __name__ == '__main__':
    SimulatorRunner.run(target_path=os.path.join(os.path.dirname(__file__), "results", "run_answer.json"))
    # SimulatorRunner.run_sensitivity(target_path_csv=os.path.join(os.path.dirname(__file__), "results", "run_sensitivity_tax_rate.csv"),       target_path_png=os.path.join(os.path.dirname(__file__), "results", "run_sensitivity_tax_rate.png"))
