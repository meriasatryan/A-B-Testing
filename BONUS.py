import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from loguru import logger
from multiprocessing import Pool
import seaborn as sns

# Set up the logger
logger.add("experiment.log", format="{time} {level} {message}", level="DEBUG")

#--------------------------------------#
class Bandit(ABC):
    """
    Abstract base class for Bandit algorithms.
    """

    @abstractmethod
    def __init__(self, p):
        """
        Initialize the bandit algorithm with the provided probabilities.

        :param p: List of probabilities representing the success rates for each bandit.
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
        :param reward: Reward received from the bandit.
        """
        pass

    @abstractmethod
    def experiment(self, num_trials, dynamic_p=None):
        """
        Run the experiment for a specified number of trials.

        :param num_trials: Total number of trials to run.
        :param dynamic_p: Optional function for dynamic bandit probabilities.
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
class EpsilonGreedy(Bandit):
    """
    Epsilon-Greedy algorithm for the multi-armed bandit problem.
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
        self.counts = np.zeros(self.n_bandits)
        self.rewards = np.zeros(self.n_bandits)
        self.cumulative_rewards = []
        self.cumulative_regret = []

    def pull(self):
        """
        Select a bandit to pull using epsilon-greedy logic.

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

    def experiment(self, num_trials, dynamic_p=None):
        """
        Run the Epsilon-Greedy experiment for a given number of trials.

        :param num_trials: Total number of trials to simulate.
        :param dynamic_p: Optional function for dynamic bandit probabilities.
        """
        optimal_bandit = np.argmax(self.p)
        for t in range(1, num_trials + 1):
            if dynamic_p:
                self.p = dynamic_p(t)  # Update probabilities dynamically

            self.epsilon = 1 / t  # Decay epsilon
            bandit_idx = self.pull()
            reward = np.random.rand() < self.p[bandit_idx]
            self.update(bandit_idx, reward)

            # Track cumulative metrics
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

        results_df = pd.DataFrame({
            "Trial": np.arange(1, len(self.cumulative_rewards) + 1),
            "Cumulative Reward": self.cumulative_rewards,
            "Cumulative Regret": self.cumulative_regret
        })
        results_df.to_csv("epsilon_greedy_results_bonus.csv", index=False)

        logger.info("Epsilon Greedy Report (Bonus Implementation):")
        logger.info(f"- Total Cumulative Reward: {self.cumulative_rewards[-1]:.2f}")
        logger.info(f"- Average Reward per Trial: {avg_reward:.4f}")
        logger.info(f"- Final Cumulative Regret: {self.cumulative_regret[-1]:.2f}")
        logger.info(f"- Average Regret per Trial: {avg_regret:.4f}")

#--------------------------------------#
def dynamic_probabilities(t):
    """
    Generate dynamic success probabilities for bandits over time.

    :param t: Current trial number.
    :return: List of probabilities for each bandit.
    """
    return [
        0.1 + 0.05 * np.sin(0.1 * t),
        0.3 + 0.05 * np.cos(0.1 * t),
        0.5 - 0.05 * np.sin(0.1 * t),
        0.7 - 0.05 * np.cos(0.1 * t)
    ]

#--------------------------------------#
def parallel_experiment(algorithm, p, num_trials, dynamic_p=None):
    """
    Run the experiment in parallel for faster execution.

    :param algorithm: Name of the bandit algorithm (e.g., "EpsilonGreedy").
    :param p: Initial probabilities for the bandits.
    :param num_trials: Number of trials to simulate.
    :param dynamic_p: Optional dynamic probabilities function.
    :return: Tuple of cumulative rewards and regrets.
    """
    bandit = EpsilonGreedy(p) if algorithm == "EpsilonGreedy" else None  # Extend for more algorithms
    bandit.experiment(num_trials, dynamic_p)
    return bandit.cumulative_rewards, bandit.cumulative_regret

#--------------------------------------#
if __name__ == '__main__':
    """
    Main execution block for running the enhanced A/B Testing Experiment.
    """
    logger.info("Starting the Enhanced A/B Testing Experiment")

    # Run parallel experiments
    with Pool(processes=2) as pool:
        results = pool.starmap(parallel_experiment, [
            ("EpsilonGreedy", [1, 2, 3, 4], 20000, dynamic_probabilities),
            ("EpsilonGreedy", [1, 2, 3, 4], 20000)  
        ])

    # Extract results
    rewards_dynamic, regrets_dynamic = results[0]
    rewards_static, regrets_static = results[1]

    # Final results for Dynamic Probabilities
    total_rewards_dynamic = rewards_dynamic[-1]
    avg_rewards_dynamic = np.mean(rewards_dynamic)
    final_regret_dynamic = regrets_dynamic[-1]
    avg_regret_dynamic = np.mean(regrets_dynamic)

    # Final results for Static Probabilities
    total_rewards_static = rewards_static[-1]
    avg_rewards_static = np.mean(rewards_static)
    final_regret_static = regrets_static[-1]
    avg_regret_static = np.mean(regrets_static)

    # Print Summary
    print("\n--- Final Results ---")
    print("\nDynamic Probabilities:")
    print(f"Total Cumulative Reward: {total_rewards_dynamic:.2f}")
    print(f"Average Reward per Trial: {avg_rewards_dynamic:.4f}")
    print(f"Final Cumulative Regret: {final_regret_dynamic:.2f}")
    print(f"Average Regret per Trial: {avg_regret_dynamic:.4f}")

    print("\nStatic Probabilities:")
    print(f"Total Cumulative Reward: {total_rewards_static:.2f}")
    print(f"Average Reward per Trial: {avg_rewards_static:.4f}")
    print(f"Final Cumulative Regret: {final_regret_static:.2f}")
    print(f"Average Regret per Trial: {avg_regret_static:.4f}")

    # Visualization
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=np.arange(20000), y=rewards_dynamic, label="Dynamic Probabilities")
    sns.lineplot(x=np.arange(20000), y=rewards_static, label="Static Probabilities")
    plt.title("Cumulative Rewards Comparison")
    plt.xlabel("Trials")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.show()

    logger.info("Enhanced Experiment Completed Successfully")