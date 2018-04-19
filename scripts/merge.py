"""
resyncs logs from server, then plots graph

needs config file ~/instances.yaml
"""
import argparse
import yaml
import subprocess
import os
from os import path
import plot_graphs


def run(hostname, logfile, **kwargs):
    with open('~/instances.yaml'.replace('~', os.environ['HOME']), 'r') as f:
        config = yaml.load(f)
    ip_address = config['ip_by_name'][hostname]
    local_path = os.getcwd() + '/logs'
    remote_path = local_path.replace(os.environ['HOME'], '/home/ubuntu')

    # this is some hacky stuff for my own environment, it'll just get ignored for
    # your own environment:
    if os.environ.get('ROUTING_COMMAND', '') != '':
        cmd_list = [
            os.environ['ROUTING_COMMAND'], ip_address
        ]
        print(cmd_list)
        print(subprocess.check_output(cmd_list).decode('utf-8'))

    cmd_list = [
        'rsync', '-av',
        '-e', 'ssh -i %s' % config['keyfile'].replace('~/', os.environ['HOME'] + '/'),
        'ubuntu@{ip_address}:{remote_path}/'.format(remote_path=remote_path, ip_address=ip_address),
        '{local_path}/'.format(local_path=local_path)
    ]
    print(cmd_list)
    print(subprocess.check_output(cmd_list).decode('utf-8'))
    if logfile is None:
        files = os.listdir('logs')
        logfile = 'logs/' + sorted(files)[-1]
        print(logfile)
    plot_graphs.plot_reward(logfile=logfile, **kwargs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hostname', type=str, help='should be in hosts.yaml', required=True)
    parser.add_argument('--logfile', type=str)
    parser.add_argument('--max-x', type=int)
    parser.add_argument('--min-y', type=float)
    parser.add_argument('--max-y', type=float)
    parser.add_argument('--title', type=str)
    args = parser.parse_args()
    run(**args.__dict__)
