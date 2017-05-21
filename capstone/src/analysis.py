import numpy as np
import glob
import seaborn as sns
import os

saves = os.listdir('results')
print(saves)
saves.sort()
print(saves)


def moving_average(data, window=100):
    avgs = np.empty_like(data)
    for i in range(data.shape[0]):
        avgs[i] = np.mean(data[i:i+100])
    return avgs


def load_data(directory, keep_test=False):
    files = glob.glob('results/' + directory + '/*_creward.npy')
    files.sort()
    reward_data = np.empty(len(files) * 10)
    reward_data.fill(np.NaN)
    for i, f in enumerate(files):
        d = np.load(f)
        reward_data[i * 10: i * 10 + d.shape[0]] = d
    reward_data = reward_data[~np.isnan(reward_data)]

    if not keep_test:
        reward_data = reward_data[:-200]

    files = glob.glob('results/' + directory + '/*_episode_duration.npy')
    files.sort()
    duration_data = np.empty(len(files) * 10)
    duration_data.fill(np.NaN)
    for i, f in enumerate(files):
        d = np.load(f)
        duration_data[i * 10: i * 10 + d.shape[0]] = d
    duration_data = duration_data[~np.isnan(duration_data)]

    if not keep_test:
        duration_data = duration_data[:-200]

    return reward_data, duration_data


def compute_total_time(duration_data):
    return np.sum(duration_data)


if __name__ == '__main__':
    for i, configuration in enumerate(saves):
        reward, duration = load_data(configuration)
        total_time = np.sum(duration)
        total_iterations = duration.shape[0]
        print(
        'Configuration #{} finished after {:2.4} minutes and {} iterations. It/min: {:2.4}'.format(i, total_time / 60.0,
                                                                                                   total_iterations,
                                                                                                   total_iterations * 1.0 / (
                                                                                                   total_time / 60)))
        sns.tsplot(reward)
        sns.plt.title("#{} configuration reward".format(i))
        sns.plt.legend(['Iteration reward'])
        sns.plt.xlabel('Iteration number')
        sns.plt.ylabel('Reward')
        sns.plt.savefig('../parameter_exploration/parameter_set_{}.png'.format(i))
        sns.plt.clf()
        # sns.plt.show()
