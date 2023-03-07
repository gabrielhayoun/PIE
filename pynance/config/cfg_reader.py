from configobj import ConfigObj, flatten_errors
from validate import Validator
from pathlib import Path
import numpy as np
from datetime import datetime

# local imports
from pynance.utils.user import get_path_to_data

""" *read* is a function that reads a config from a file and convert every field in the right format and fill the necessary not given fields.  
    Please refer to : http://www.voidspace.org.uk/python/articles/configobj.shtml
    for a detailed explanation of ConfigObj.
"""

# TODO: can we give spec file name in param instead ??
# good idea only for "Never" changing again the cfg_read.py function
# however it is very necessary to adapt the code that uses the parameters
# as returns by _read function

train_spec_filename = str((Path(__file__).resolve().parents[0]/'spec_train.cfg').resolve())  
infer_spec_filename = str((Path(__file__).resolve().parents[0]/'spec_infer.cfg').resolve())  
crypto_spec_filename = str((Path(__file__).resolve().parents[0]/'spec_crypto.cfg').resolve())  

def read(file_name, kind):
    if(kind == 'train'):
        spec_file = train_spec_filename
    elif(kind == 'infer'):
        spec_file = infer_spec_filename
    elif(kind == 'crypto'):
        spec_file = crypto_spec_filename
    else:
        raise ValueError(f'Not recognized kind {kind}. Available: train, infer, crypto.')

    return _read(file_name, spec_file)

def _read(filename, spec_file):
    # loading the config spec
    configspec = ConfigObj(spec_file, interpolation=False, list_values=False,
                           _inspec=True)
    # loading the current filename
    config = ConfigObj(str(filename), configspec=configspec)
    # validating it
    validator = Validator({
        'actions_file_name': check_actions_file_name,
        'date': check_date
        })

    results = config.validate(validator)
    if results != True:
        for (section_list, key, _) in flatten_errors(config, results):
            if key is not None:
                print ('The "%s" key in the section "%s" failed validation' % (key, ', '.join(section_list)))
            else:
                if(len(section_list)==1):
                    print ('The following section was missing: %s ' % ', '.join(section_list))
                else:
                    print ('The following sections were missing: %s ' % ', '.join(section_list))
    return config

# -------------- custom check -------------------- #
# TODO : the default value should not be given there
def check_actions_file_name(value):
    return get_path_to_data() / value

def check_date_format(value):
    format = "%Y-%m-%d"
    try:
        res = bool(datetime.strptime(value, format))
    except ValueError:
        res = False
        raise ValueError(f'Date format is wrong. Should be: YYYY-MM-DD. Got: {value}.')
    return res

def check_date(value):
    if(value == 'None'):
        return None
    else: # '1999-01-01'
        check_date_format(value)
    return value
