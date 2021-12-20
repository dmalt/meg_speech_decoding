import numpy as np
import math


MODEL_DUMPS_DIR = 'model_dumps'
RESULTS_DIR = 'results'


REGRESSION_MODE = "regression"
CLASSIFICATION_MODE = "classification"

WORDS_REMAP = {
    'silent' : 0,

    'женя' : 1,
    'широко' : 2,
    'шагает' : 3,
    'желтых' : 4,
    'штанах' : 5,

    'шуру' : 6,
    'ужалил' : 7,
    'шершень' : 8,

    'лара' : 9,
    'ловко' : 10,
    'крутит' : 11,
    'руль' : 12,
    'левой' : 13,
    'рукой' : 14,

    'лирику' : 15,
    'любит' : 16,
    'лиля' : 17,

    'бабушка' : 18,
    'боится' : 19,
    'барабанов' : 20,

    'белого' : 21,
    'барана' : 22,
    'больно' : 23,
    'бодает' : 24,
    'бешеный' : 25,
    'бык' : 26
}


def data_generator(X, Y, batch_size, lag_backward, lag_forward, shuffle=True, infinite=True):
    assert len(X)==len(Y) or len(Y)==0
    total_lag = lag_backward + lag_forward
    all_batches = math.ceil((X.shape[0] - total_lag) / batch_size)
    samples_in_last_batch = (X.shape[0] - total_lag) % batch_size
    batch = 0
    random_core = np.arange(lag_backward, X.shape[0] - lag_forward)
    while True:
        if shuffle:
            np.random.shuffle(random_core)
        for batch in range(all_batches):       
            batch_start = batch * batch_size
            batch_end = (batch + 1) * batch_size
            if batch_end >= len(random_core):
                batch_end = None
            batch_samples = random_core[batch_start : batch_end]

            batch_x = np.array([X[i - lag_backward : i + lag_forward + 1] for i in batch_samples])
            batch_x = np.swapaxes(batch_x, 1, 2)

            if len(Y) > 0:
                batch_y = Y[[batch_samples]] 
                yield (batch_x, batch_y)
            else:
                yield batch_x
        
        if not infinite:
            break