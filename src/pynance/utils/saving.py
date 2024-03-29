from configobj import ConfigObj

# saving the new params to a file so the user can go and debug it or refer to all the simulations params when needed.
def save_configobj(parameters, saving_dir, name='parameters'):
    if(type(parameters) != dict):
        parameters = parameters.dict()
    pp_dict = convert_objects(parameters)
    pp = ConfigObj(pp_dict)
    pp.filename = '{}/{}.cfg'.format(saving_dir, name)
    pp.write()

# ----------------- convert ------------------- #
def convert_objects(p):
    pp = {}
    for k, v in p.items():
        if(type(v) is dict):
            pp[k] = convert_objects(v)
        elif(type(v) is list):
            for i in range(len(v)):
                v[i] = convert_object(v[i])
            pp[k] = v
        else:
            pp[k] = convert_object(v)
    return pp

def convert_object(o):
    try :
        return '{}{}'.format(o.__name__, inspect.signature(o))
    except Exception:
        try :
            return o.__str__()
        except Exception:
            return o
