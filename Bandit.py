import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from loguru import logger

# Set up the logger
logger.add("experiment.log", format="{time} {level} {message}", level="DEBUG")

#--------------------------------------#
class Bandit(ABC):
    """
    Abstract base class for Bandit algorithms.
    
    This class outlines the structure for implementing different bandit algorithms.
    """

    @abstractmethod
    def __init__(self, p):
        """
        Initialize the bandit algorithm with the provided probabilities.

        :param p: List of probabilities representing the success rates for each bandit.
        """
        pass

    @abstractmethod
    def __repr__(self):
        """
        Return a string representation of the bandit algorithm.
        
        :return: String representation of the algorithm.
        """
        pass

    @abstractmethod
    def pull(self):
        """
        Select a bandit to pull (action).

        :return: Index of the selected bandit.
        """
        pass

    @abstractmethod
    def update(self, bandit_idx, reward):
        """
        Update the algorithm's state based on the bandit pulled and the reward received.

        :param bandit_idx: Index of the bandit that was pulled.
        :param reward: Reward received from the pulled bandit.
        """
        pass

    @abstractmethod
    def experiment(self, num_trials):
        """
        Run the experiment for a specified number of trials.

        :param num_trials: Total number of trials to run.
        """
        pass

    @abstractmethod
    def report(self):
        """
        Log the results and store the data in a CSV file.

        Outputs cumulative rewards, average rewards, and regrets.
        """
        pass

#--------------------------------------#
class Visualization:
    """
    Helper class for visualizing results from the bandit experiments.
    """

    def plot1(self, rewards1, rewards2, title="Learning Process"):
        """
        Plot cumulative rewards for two bandit algorithms.

        :param rewards1: List of cumulative rewards from the first algorithm.
        :param rewards2: List of cumulative rewards from the second algorithm.
        :param title: Title for the plot. Default is "Learning Process".
        """
        plt.plot(rewards1, label="Epsilon-Greedy")
        plt.plot(rewards2, label="Thompson Sampling")
        plt.title(title)
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Reward")
        plt.legend()
        plt.show()

    def plot2(self, regrets1, regrets2, title="Cumulative Regret"):
        """
        Plot cumulative regret for two bandit algorithms.

        :param regrets1: List of cumulative regrets from the first algorithm.
        :param regrets2: List of cumulative regrets from the second algorithm.
        :param title: Title for the plot. Default is "Cumulative Regret".
        """
        plt.plot(regrets1, label="Epsilon-Greedy")
        plt.plot(regrets2, label="Thompson Sampling")
        plt.title(title)
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Regret")
        plt.legend()
        plt.show()

#--------------------------------------#
class EpsilonGreedy(Bandit):
    """
    Epsilon-Greedy algorithm for the multi-armed bandit problem.

    This algorithm balances exploration and exploitation by using a decaying epsilon value.
    """

    def __init__(self, p, epsilon_decay=0.01):
        """
        Initialize the Epsilon-Greedy algorithm.

        :param p: List of success probabilities for each bandit.
        :param epsilon_decay: Rate of decay for epsilon. Default is 0.01.
        """
        self.p = p
        self.n_bandits = len(p)
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_decay = epsilon_decay
        self.counts = np.zeros(self.n_bandits)  # Number of pulls per bandit
        self.rewards = np.zeros(self.n_bandits)  # Total rewards per bandit
        self.cumulative_rewards = []  # Cumulative rewards over time
        self.cumulative_regret = []  # Cumulative regret over time

    def __repr__(self):
        """
        Return a string representation of the Epsilon-Greedy algorithm.

        :return: String representation of the algorithm.
        """
        return f"EpsilonGreedy(epsilon={self.epsilon})"

    def pull(self):
        """
        Select a bandit to pull based on epsilon-greedy logic.

        :return: Index of the selected bandit.
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_bandits)  # Explore
        else:
            return np.argmax(self.rewards / (self.counts + 1e-10))  # Exploit

    def update(self, bandit_idx, reward):
        """
        Update the state of the algorithm based on the bandit pulled and reward received.

        :param bandit_idx: Index of the bandit pulled.
        :param reward: Reward received from the bandit.
        """
        self.counts[bandit_idx] += 1
        self.rewards[bandit_idx] += reward

    def experiment(self, num_trials):
        """
        Run the Epsilon-Greedy experiment for a given number of trials.

        :param num_trials: Total number of trials to simulate.
        """
        optimal_bandit = np.argmax(self.p)
        for t in range(1, num_trials + 1):
            self.epsilon = 1 / t  # Decay epsilon
            bandit_idx = self.pull()
            reward = np.random.rand() < self.p[bandit_idx]
            self.update(bandit_idx, reward)

            # Track cumulative rewards and regret
            self.cumulative_rewards.append(np.sum(self.rewards))
            regret = np.sum(self.p[optimal_bandit] * t) - np.sum(self.rewards)
            self.cumulative_regret.append(regret)

            logger.debug(f"Trial {t}: Bandit {bandit_idx}, Reward {reward}, Epsilon {self.epsilon}")

    def report(self):
        """
        Log the results and store cumulative rewards and regret in a CSV file.

        Outputs the total cumulative reward, average reward, and final cumulative regret.
        """
        avg_reward = np.mean(self.cumulative_rewards)
        avg_regret = np.mean(self.cumulative_regret)

        # Save results to a CSV file
        results_df = pd.DataFrame({
            "Trial": np.arange(1, len(self.cumulative_rewards) + 1),
            "Cumulative Reward": self.cumulative_rewards,
            "Cumulative Regret": self.cumulative_regret
        })
        results_df.to_csv("epsilon_greedy_results.csv", index=False)

        # Log summary statistics
        logger.info("Epsilon Greedy Report:")
        logger.info(f"- Total Cumulative Reward: {self.cumulative_rewards[-1]:.2f}")
        logger.info(f"- Average Reward per Trial: {avg_reward:.4f}")
        logger.info(f"- Final Cumulative Regret: {self.cumulative_regret[-1]:.2f}")
        logger.info(f"- Average Regret per Trial: {avg_regret:.4f}")

class ThompsonSampling(Bandit):
    """
    Thompson Sampling algorithm for the multi-armed bandit problem.

    This algorithm uses a Bayesian approach to balance exploration and exploitation
    by sampling from a Beta distribution for each bandit.
    """

    def __init__(self, p):
        """
        Initialize the Thompson Sampling algorithm.

        :param p: List of success probabilities for each bandit.
        """
        self.p = p
        self.n_bandits = len(p)
        self.successes = np.zeros(self.n_bandits)  # Number of successes for each bandit
        self.failures = np.zeros(self.n_bandits)  # Number of failures for each bandit
        self.cumulative_rewards = []  # Cumulative rewards over time
        self.cumulative_regret = []  # Cumulative regret over time

    def __repr__(self):
        """
        Return a string representation of the Thompson Sampling algorithm.

        :return: String representation of the algorithm.
        """
        return f"ThompsonSampling()"

    def pull(self):
        """
        Select a bandit to pull based on samples from the Beta distribution.

        :return: Index of the selected bandit.
        """
        samples = [np.random.beta(1 + self.successes[i], 1 + self.failures[i]) for i in range(self.n_bandits)]
        return np.argmax(samples)

    def update(self, bandit_idx, reward):
        """
        Update the state of the algorithm based on the bandit pulled and reward received.

        :param bandit_idx: Index of the bandit pulled.
        :param reward: Reward received from the bandit.
        """
        if reward:
            self.successes[bandit_idx] += 1
        else:
            self.failures[bandit_idx] += 1

    def experiment(self, num_trials):
        """
        Run the Thompson Sampling experiment for a given number of trials.

        :param num_trials: Total number of trials to simulate.
        """
        optimal_bandit = np.argmax(self.p)
        for t in range(1, num_trials + 1):
            bandit_idx = self.pull()
            reward = np.random.rand() < self.p[bandit_idx]
            self.update(bandit_idx, reward)

            # Track cumulative rewards and regret
            self.cumulative_rewards.append(np.sum(self.successes))
            regret = np.sum(self.p[optimal_bandit] * t) - np.sum(self.successes)
            self.cumulative_regret.append(regret)

            logger.debug(f"Trial {t}: Bandit {bandit_idx}, Reward {reward}")

    def report(self):
        """
        Log the results and store cumulative rewards and regret in a CSV file.

        Outputs the total cumulative reward, average reward, and final cumulative regret.
        """
        avg_reward = np.mean(self.cumulative_rewards)
        avg_regret = np.mean(self.cumulative_regret)

        # Save results to a CSV file
        results_df = pd.DataFrame({
            "Trial": np.arange(1, len(self.cumulative_rewards) + 1),
            "Cumulative Reward": self.cumulative_rewards,
            "Cumulative Regret": self.cumulative_regret
        })
        results_df.to_csv("thompson_sampling_results.csv", index=False)

        # Log summary statistics
        logger.info("Thompson Sampling Report:")
        logger.info(f"- Total Cumulative Reward: {self.cumulative_rewards[-1]:.2f}")
        logger.info(f"- Average Reward per Trial: {avg_reward:.4f}")
        logger.info(f"- Final Cumulative Regret: {self.cumulative_regret[-1]:.2f}")
        logger.info(f"- Average Regret per Trial: {avg_regret:.4f}")

#--------------------------------------#
def comparison():
    """
    Compare the performance of Epsilon-Greedy and Thompson Sampling algorithms.

    This function initializes both algorithms, runs experiments, and generates
    visualizations for cumulative rewards and regrets. It also logs results to
    a file and saves experiment data in CSV format.
    """
    p = [1, 2, 3, 4]  # Success probabilities for each bandit
    num_trials = 20000  # Total number of trials

    # Initialize algorithms
    eg = EpsilonGreedy(p)
    ts = ThompsonSampling(p)

    # Run experiments
    eg.experiment(num_trials)
    ts.experiment(num_trials)

    # Generate visualizations
    vis = Visualization()
    vis.plot1(eg.cumulative_rewards, ts.cumulative_rewards, title="Cumulative Rewards")
    vis.plot2(eg.cumulative_regret, ts.cumulative_regret, title="Cumulative Regrets")

    # Generate and log reports
    eg.report()
    ts.report()

#--------------------------------------#
if __name__ == '__main__':
    """
    Main execution block for running the A/B Testing Experiment.

    This block initializes the logger, runs the comparison function, and logs
    the completion of the experiment.
    """
    logger.info("Starting the A/B Testing Experiment")
    comparison()  # Run the comparison function
    logger.info("Experiment Completed Successfully")
