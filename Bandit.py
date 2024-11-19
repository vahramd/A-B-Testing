import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from loguru import logger

# Define output directory
OUTPUT_DIR = "/Users/vahramdressler/Desktop"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize logger
logger.add(os.path.join(OUTPUT_DIR, "experiment.log"), level="INFO", format="{time} {level} {message}")

# =========================================
# Bandit Abstract Class
# =========================================
class Bandit(ABC):
    def __init__(self, p, name):
        self.p = p
        self.name = name
        self.successes = 0
        self.failures = 0
        self.n = 0
        self.rewards = []

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self, reward):
        pass

    def save_to_csv(self, algorithm_name):
        data = {"Trial": list(range(1, len(self.rewards) + 1)), "Reward": self.rewards}
        df = pd.DataFrame(data)
        filename = os.path.join(OUTPUT_DIR, f"{algorithm_name}_{self.name}.csv")
        df.to_csv(filename, index=False)
        logger.info(f"Results saved to {filename}")

    def plot_learning_process(self, algorithm_name):
        cumulative_average = np.cumsum(self.rewards) / (np.arange(1, len(self.rewards) + 1))
        plt.figure(figsize=(8, 5))
        plt.plot(cumulative_average, label=f"{self.name}")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Average Reward")
        plt.title(f"{algorithm_name} - {self.name}")
        plt.legend()
        plt.grid()
        plot_path = os.path.join(OUTPUT_DIR, f"{algorithm_name}_{self.name}.png")
        plt.savefig(plot_path)
        logger.info(f"Plot saved for {self.name} under {algorithm_name} at {plot_path}")
        plt.close()

# =========================================
# Epsilon-Greedy Algorithm
# =========================================
class EpsilonGreedy(Bandit):
    def __init__(self, p, epsilon, name):
        super().__init__(p, name)
        self.epsilon = epsilon
        self.estimated_mean = 0

    def pull(self):
        return 1 if np.random.rand() < self.p else 0

    def update(self, reward):
        self.n += 1
        self.rewards.append(reward)
        self.estimated_mean += (reward - self.estimated_mean) / self.n

    def experiment(self, num_trials, bandits):
        for t in range(1, num_trials + 1):
            self.epsilon = 1 / t
            if np.random.rand() < self.epsilon:
                choice = np.random.randint(len(bandits))
            else:
                choice = np.argmax([b.estimated_mean for b in bandits])

            reward = bandits[choice].pull()
            bandits[choice].update(reward)

# =========================================
# Thompson Sampling Algorithm
# =========================================
class ThompsonSampling(Bandit):
    def __init__(self, p, name):
        super().__init__(p, name)
        self.alpha = 1
        self.beta = 1

    def pull(self):
        return 1 if np.random.rand() < self.p else 0

    def update(self, reward):
        self.n += 1
        self.rewards.append(reward)
        if reward:
            self.alpha += 1
        else:
            self.beta += 1

    def experiment(self, num_trials, bandits):
        for _ in range(num_trials):
            samples = [np.random.beta(b.alpha, b.beta) for b in bandits]
            choice = np.argmax(samples)
            reward = bandits[choice].pull()
            bandits[choice].update(reward)

# =========================================
# Main Experiment
# =========================================
if __name__ == "__main__":
    num_trials = 20000
    bandit_probabilities = [0.1, 0.3, 0.5, 0.7]

    # Run Epsilon-Greedy Experiment
    logger.info("Starting Epsilon-Greedy Experiment...")
    bandits_eg = [EpsilonGreedy(p, epsilon=0.1, name=f"Bandit-{i+1}") for i, p in enumerate(bandit_probabilities)]
    eg_algorithm = EpsilonGreedy(0, 0.1, "Epsilon-Greedy")
    eg_algorithm.experiment(num_trials, bandits_eg)
    for b in bandits_eg:
        b.plot_learning_process("Epsilon-Greedy")
        b.save_to_csv("Epsilon-Greedy")

    # Run Thompson Sampling Experiment
    logger.info("Starting Thompson Sampling Experiment...")
    bandits_ts = [ThompsonSampling(p, name=f"Bandit-{i+1}") for i, p in enumerate(bandit_probabilities)]
    ts_algorithm = ThompsonSampling(0, "Thompson-Sampling")
    ts_algorithm.experiment(num_trials, bandits_ts)
    for b in bandits_ts:
        b.plot_learning_process("Thompson Sampling")
        b.save_to_csv("Thompson Sampling")

    logger.info("Experiments completed successfully.")
