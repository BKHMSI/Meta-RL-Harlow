import os
import numpy as np 
import matplotlib.pyplot as plt

if __name__ == "__main__":

    all_rewards = []

    base_path = "ckpt"
    run_title = "Harlow_Final_LSTM"
    n_seeds = 8
    n_workers = 8
    for seed in range(1, n_seeds+1):
        run = run_title + f"_{seed}"
        run_rewards = []
        for worker in range(n_workers):
            path = os.path.join(base_path, run, f"rewards_{worker}.npy")
            if os.path.exists(path):
                rewards = np.load(path)
                run_rewards += [rewards[:2500]]
        all_rewards += [np.array(run_rewards).mean(axis=0)]

    all_rewards = np.stack(all_rewards)

    quantiles = [0, 500, 1000, 1500, 2000, 2500]
    n_quantiles = len(quantiles)-1
    n_trials = all_rewards.shape[2]
    for i in range(n_quantiles):
        line = []
        stds = []
        for j in range(n_trials):
            q = all_rewards[:,quantiles[i]:quantiles[i+1],j]
            performance = q.mean(axis=1)
            line += [performance.mean()*100]
            stds += [(performance.std()*100)]
        plt.errorbar(np.arange(1,7), line, fmt='o-', yerr=stds)

    plt.plot([1,6], [50,50], '--')
    plt.xlabel("Trial")
    plt.ylabel("Performance (%)")
    plt.legend(["Random", "1st", "2nd", "3rd", "4th", "Final"], title="Training Quantile")
    plt.title("Harlow Task")
    plt.show()
