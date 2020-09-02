import os
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 

from tqdm import tqdm 
from tensorboard.backend.event_processing import event_accumulator

def read_data(load_dir, tag="perf/avg_reward_100"):

    events = os.listdir(load_dir)
    for event in events:
        path = os.path.join(load_dir, event)
        ea = event_accumulator.EventAccumulator(path, size_guidance={ 
                event_accumulator.COMPRESSED_HISTOGRAMS: 0,
                event_accumulator.IMAGES: 0,
                event_accumulator.AUDIO: 0,
                event_accumulator.SCALARS: 2500,
                event_accumulator.HISTOGRAMS: 0,
        })
        
        ea.Reload()
        tags = ea.Tags()

        if tag not in tags["scalars"]: continue

        if len(ea.Scalars(tag)) == 2500:
            return np.array([s.value for s in ea.Scalars(tag)])

    return None 

def plot_rewards_curve(save_path, 
    load_path_gru, 
    load_path_rgu, 
    load_path_lstm,
    n_seeds=8,
    n_workers=8,
):

    gru_data = np.zeros((n_seeds, 2500))
    rgu_data = np.zeros((n_seeds, 2500))
    lstm_data = np.zeros((n_seeds, 2500))
    count = 0
    for seed_idx in tqdm(range(n_seeds)):
        gru_workers = []
        rgu_workers = []
        lstm_workers = []
        for worker in range(n_workers):
            gru_event = read_data(load_dir=load_path_gru+f"_{seed_idx+1}_{worker}")
            rgu_event = read_data(load_dir=load_path_rgu+f"_{seed_idx+1}_{worker}")
            lstm_event = read_data(load_dir=load_path_lstm+f"_{seed_idx+1}_{worker}")
            if not(gru_event is None or lstm_event is None or rgu_event is None): 
                gru_workers += [gru_event]
                rgu_workers += [rgu_event]
                lstm_workers += [lstm_event]
            else:
                count += 1

        gru_data[seed_idx] = np.array(gru_workers).mean(axis=0) 
        rgu_data[seed_idx] = np.array(rgu_workers).mean(axis=0) 
        lstm_data[seed_idx] = np.array(lstm_workers).mean(axis=0) 

    data = []
    for seed_idx in range(n_seeds):
        for i in range(2500):
            data += [{'Episode': i, 'Reward': gru_data[seed_idx][i], "RNN Type": "GRU"}]
            data += [{'Episode': i, 'Reward': rgu_data[seed_idx][i], "RNN Type": "RGU"}]
            data += [{'Episode': i, 'Reward': lstm_data[seed_idx][i], "RNN Type": "LSTM"}]
    
    df = pd.DataFrame(data)
    sns.lineplot(x="Episode", y="Reward", hue="RNN Type", data=df)
    plt.show()


if __name__ == "__main__":
    
    plot_rewards_curve(
        './gru_lstm.png',
        './logs_gru/HS_GRU_2',
        './logs_rgu/HS_RGU_2',
        './logs_lstm/HS_LSTM_2'
    )