from configobj import ConfigObj, flatten_errors
from validate import Validator
from pathlib import Path
import numpy as np
""" *read* is a function that reads a config from a file and convert every field in the right format and fill the necessary not given fields.  
    Please refer to : http://www.voidspace.org.uk/python/articles/configobj.shtml
    for a detailed explanation of ConfigObj.
"""

spec_filename = str((Path(__file__).resolve().parents[0]/'spec.cfg').resolve())  

def read(filename):
    # loading the config spec
    configspec = ConfigObj(spec_filename, interpolation=False, list_values=False,
                           _inspec=True)
    # loading the current filename
    config = ConfigObj(str(filename), configspec=configspec)
    # validating it
    validator = Validator({}) # give the custom checks in the dictionnary here {'my_type': my_type_fn}
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