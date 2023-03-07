""" Main function to launch a training / inference from the command line using the configuration files.
"""

import sys, getopt
from pathlib import Path

# https://www.tutorialspoint.com/python/python_command_line_arguments.htm
from pynance import train
from pynance.utils.user import get_path_to_config_files

def print_usage():
    print('Usage : ')
    print('run.py -n <cfg_name>')
    print('`.cfg` is automatically added to the name')
    print(f'File <cfg_name>.cfg should exist in folder {get_path_to_config_files()}')
    print('To change it, modify `USERCFG` file.')

def run(argv):
    try:
        opts, args = getopt.getopt(argv,"hn:", ['path=']) # h for help, s for save
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
        cfg_path = get_path_to_config_files() / f'{cfg_name}.cfg'
        print(f'Config file used: {cfg_path}')
        train.main(cfg_path)
    else :
        print('Please precise path to cfg.')
        print_usage()
        sys.exit()

if __name__ == "__main__":
    run(sys.argv[1:]) # because the first arg is always the name of the file