""" Main function to launch a training / inference from the command line using the configuration files.
"""

import sys, getopt
from pathlib import Path

# https://www.tutorialspoint.com/python/python_command_line_arguments.htm
import pynance

def print_usage():
    print('Usage : ')
    print('run.py -n <cfg_name> -k <process_kind>')
    print('`.cfg` is automatically added to the name')
    print(f'File <cfg_name>.cfg should exist in folder {pynance.utils.user.get_path_to_config_files()}')
    print('To change it, modify `USERCFG` file.')
    print('Available kind process: train, infer, crypto, coint.')

def get_function_from_kind(kind):
    switch = {
        'train': pynance.train.main,
        'infer': pynance.infer.main,
        'coint': pynance.coint.main
        # 'crypto': pynance.crypto.main
    }
    return switch.get(kind, "Invalid input")

def run(argv):
    try:
        opts, args = getopt.getopt(argv,"hn:k:", ['path=']) # h for help, s for save
    except getopt.GetoptError:
        print_usage()
        sys.exit(2)

    if(len(opts)==0):
        print_usage()
        sys.exit()

    if(opts[0][0] == '-h'):
        print_usage()
        sys.exit()
    elif opts[0][0] in ('-n','--name'):
        cfg_name = opts[0][1]
        cfg_path = pynance.utils.user.get_path_to_config_files() / f'{cfg_name}.cfg'
        print(f'Config file used: {cfg_path}')
        if(opts[1][0] in ('-k', '--kind')):
            kind = opts[1][1]
            print(f'Process kind: {kind}')
            main_fn = get_function_from_kind(kind)
            main_fn(cfg_path)
    else :
        print('Please precise path to cfg.')
        print_usage()
        sys.exit()

if __name__ == "__main__":
    run(sys.argv[1:]) # because the first arg is always the name of the file