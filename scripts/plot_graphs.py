"""
Given a logfile, plot a graph
"""
import matplotlib.pyplot as plt
import json
import argparse
import numpy as np


def plot_reward(logfile, title, min_y, max_y, max_x):
    """
    logfiles separated by : are combined
    logfiles separated by , go in separate plots
    (: binds tighter than ,)
    """
    logfiles = logfile
    split_logfiles = logfiles.split(',')
    for j, logfile_groups in enumerate(split_logfiles):
        epoch = []
        reward = []
        test_reward = []
        for logfile in logfile_groups.split(':'):
            with open(logfile, 'r') as f:
                for n, line in enumerate(f):
                    if n == 0:
                        print(logfile, line)
                        continue  # skip first line
                    line = line.strip()
                    if line == '':
                        continue
                    d = json.loads(line)
                    if max_x is not None and d['episode'] > max_x:
                        continue
                    epoch.append(int(d['episode']))
                    reward.append(float(d['avg_reward_0']))
                    if 'test_reward' in d:
                        test_reward.append(d['test_reward'])
        print('epoch[0]', epoch[0], 'epochs[-1]', epoch[-1])
        while len(epoch) > 200:
            new_epoch = []
            new_reward = []
            new_test_reward = []
            for n in range(len(epoch) // 2):
                r = (reward[n * 2] + reward[n * 2 + 1]) / 2
                e = (epoch[n * 2] + epoch[n * 2 + 1]) // 2
                new_epoch.append(e)
                new_reward.append(r)
                if len(test_reward) > 0:
                    rt = (test_reward[n * 2] + test_reward[n * 2 + 1]) / 2
                    new_test_reward.append(rt)
            epoch = new_epoch
            reward = new_reward
            test_reward = new_test_reward
        print('epoch[0]', epoch[0], 'epochs[-1]', epoch[-1])
        if min_y is None:
            min_y = 0
        if max_y is not None:
            plt.ylim([min_y, max_y])
        suffix = ''
        if len(split_logfiles) > 0:
            suffix = ' %s' % (j + 1)
        if len(test_reward) > 0:
            plt.plot(np.array(epoch) / 1000, reward, label='train' + suffix)
            plt.plot(np.array(epoch) / 1000, test_reward, label='test' + suffix)
        else:
            plt.plot(np.array(epoch) / 1000, reward, label='reward' + suffix)
    if title is not None:
        plt.title(title)
    plt.xlabel('Episodes of 128 games (thousands)')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()
    # plt.savefig('/tmp/out-reward.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('logfile', type=str)
    parser.add_argument('--title', type=str)
    parser.add_argument('--max-x', type=int)
    parser.add_argument('--min-y', type=float)
    parser.add_argument('--max-y', type=float)

    args = parser.parse_args()
    plot_reward(**vars(args))
