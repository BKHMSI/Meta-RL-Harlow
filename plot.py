import os
import numpy as np 
import matplotlib.pyplot as plt

if __name__ == "__main__":

    all_rewards = []

    base_path = "ckpt"
    run_title = "HarlowSimple_11"
    seeds = [1, 2, 3, 4]
    for seed in seeds:
        run = run_title + f"_{seed}"
        path = os.path.join(base_path, run, run+f"_0_rewards.npy")
        if os.path.exists(path):
            rewards = np.load(path)
            all_rewards += [rewards[:2500]]

    all_rewards = np.stack(all_rewards)

    quantiles = [0, 500, 1500, 2000, 2500]
    n_quantiles = len(quantiles)-1
    n_trials = all_rewards.shape[2]
    for i in range(n_quantiles):
        line = []
        stds = []
        for j in range(n_trials):
            q = all_rewards[:,quantiles[i]:quantiles[i+1],j]
            performance = q.mean(axis=1)
            line += [performance.mean()*100]
            stds += [(performance.std()*100) / np.sqrt(all_rewards.shape[0])]
        plt.errorbar(np.arange(1,7), line, fmt='o-', yerr=stds)

    plt.plot([1,6], [50,50], '--')
    plt.xlabel("Trial")
    plt.ylabel("Performance (%)")
    plt.legend(["Random", "1st", "2nd", "3rd", "Final",], title="Training Quantile")
    plt.title("Harlow Task (Simple)")
    plt.show()