import pynance
from datetime import datetime

def get_results_dir(name, add_date=False, create_dir=True):
    if(add_date):
        now = datetime.now().strftime(format='%Y%m%d%H%M')
        results_dir = pynance.utils.user.get_path_to_results() / '{}_{}'.format(now, name)
    else:
        results_dir = pynance.utils.user.get_path_to_results() / '{}'.format(name)
    
    if(create_dir):
        results_dir.mkdir(parents=True, exist_ok=True)

    return results_dir
