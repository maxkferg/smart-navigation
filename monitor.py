#!/usr/bin/env python
"""
Monitor the positions of all the objects
Pulls object positions from Redis
"""

import os
import sys
import time
import argparse
from environments.redis.environment import ViewingEnvironment


def monitor():
    env = ViewingEnvironment(disable_render=False)

    print("Moving target")
    env.primary.target.x = 400
    env.primary.target.y = 400
    env.primary.target.save()

    while True:
        env.step(action=None)
        env.render()
        time.sleep(0.1)
        print(".", end="", flush=True)

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument('--env', help='environment ID', default='Hopper-v1')
    #parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    #parser.add_argument('--eval', help='Evaluate solution', type=bool, default=False)
    #parser.add_argument('--render', help='Render evaluation', type=bool, default=False)
    #parser.add_argument('--physics', help='Sensor for position', type=str, default='physics')
    args = parser.parse_args()
    monitor()


if __name__ == '__main__':
    main()