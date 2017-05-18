import numpy as np
import glob
import seaborn as sns
import os

saves = os.listdir('results')
print(saves)
saves.sort()
print(saves)

def load_data(directory):
    files = glob.glob('results/' + directory + '/*_creward.npy')
    files.sort()
    reward_data = np.empty(len(files) * 10)
    reward_data.fill(np.NaN)
    for i, f in enumerate(files):
        d = np.load(f)
        reward_data[i*10: i*10+d.shape[0]] = d
    reward_data = reward_data[~np.isnan(reward_data)]

    reward_data = reward_data[:-200]



    files = glob.glob('results/' + directory + '/*_episode_duration.npy')
    files.sort()
    duration_data = np.empty(len(files) * 10)
    duration_data.fill(np.NaN)
    for i, f in enumerate(files):
        d = np.load(f)
        duration_data[i*10: i*10+d.shape[0]] = d
    duration_data = duration_data[~np.isnan(duration_data)]

    duration_data = duration_data[:-200]

    return reward_data, duration_data

def compute_total_time(duration_data):
    return np.sum(duration_data)

for i, configuration in enumerate(saves):
    reward, duration = load_data(configuration)
    total_time = np.sum(duration)
    total_iterations = duration.shape[0]
    print('Configuration #{} finished after {:2.4} minutes and {} iterations. It/min: {:2.4}'.format(i, total_time/60.0,
                total_iterations, total_iterations * 1.0/(total_time/60)))
    sns.tsplot(reward)
    sns.plt.title("#{} Episode reward".format(i))
    # sns.plt.show()