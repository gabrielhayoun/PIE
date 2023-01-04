
def r_squared(y_true, y_pred):
    u = ((y_true - y_pred)** 2).sum()
    v = ((y_true - y_pred.mean()) ** 2).sum()
    return (1 - u/v)
