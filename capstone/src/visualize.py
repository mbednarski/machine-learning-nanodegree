import seaborn as sns
import numpy as np
from analysis import load_data, moving_average



reward_data, duration_data = load_data('2017-05-12_18_50_34', keep_test=True)
averages = moving_average(reward_data)
print(reward_data.shape)

sns.tsplot(reward_data)
sns.tsplot(averages, color='red')
sns.plt.title('PEPG reward per iteration on LunarLander-v2')
sns.plt.xlabel('Iteration number')
sns.plt.ylabel('Reward')
sns.plt.legend(['Iteration reward', '100 iterations average'])
sns.plt.savefig('final_plot.png')
sns.plt.show()


sns.tsplot(duration_data)
sns.plt.title('PEPG iteration duration on LunarLander-v2')
sns.plt.xlabel('Iteration number')
sns.plt.ylabel('Duration [s]')
sns.plt.legend(['Iteration duration'])
sns.plt.savefig('final_plot_duration.png')
sns.plt.show()