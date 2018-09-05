import json
import argparse
import os

EXPERIMENT_DIR = './experiments'

def get_list_params(data):
    list_params = []
    for k in data:
        if isinstance(data[k],list):
            list_params.append(k)


if __name__ == '__main__':
    # LOAD CONFIG FILE
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, help='name')
    args = parser.parse_args()

    with open(os.path.join(EXPERIMENT_DIR,args.name+'.json')) as f:
        data = json.load(f)
    