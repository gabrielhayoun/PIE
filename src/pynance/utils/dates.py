import datetime

def get_start_date(date, delta):
    format = '%Y-%m-%d'
    date = datetime.datetime.strptime(date, format)
    list_dates = make_dates(-delta, None, date)
    first_date = list_dates[0].strftime(format=format)
    return first_date

def make_dates(length_preds, init_date=None, end_date=None):
    assert(init_date is not None or end_date is not None)
    dates = []
    date = init_date
    dt = datetime.timedelta(days=1)
    if(init_date is None):
        dt = -dt
        date = end_date
    while(len(dates) < length_preds):
        date += dt
        if(date.isoweekday() <= 5):
            dates.append(date)
    if(init_date is None):
        return dates[::-1]
    return dates