import os
import numpy as np


DATA_PATH = '../norway_3g_data/cooked_data/'
PAST_LEN = 10
FUTURE_LEN = 10

def read_data():
    files = os.listdir(DATA_PATH)
    x = []
    y = []

    for file in files:
        file_path = DATA_PATH +  file
        
        time_pt = np.zeros(PAST_LEN + FUTURE_LEN)
        bw_pt = np.zeros(PAST_LEN + FUTURE_LEN)

        with open(file_path, 'rb') as f:
            for line in f:
                parse = line.split()
                time = float(parse[0])
                bw = float(parse[1])
                time_pt[:-1] = time_pt[1:]
                bw_pt[:-1] = bw_pt[1:]
                time_pt[-1] = time
                bw_pt[-1] = bw

                if time_pt[PAST_LEN] != 0:
                    x.append(np.array(bw_pt[:PAST_LEN], copy=True))
                    y.append(np.array(bw_pt[-FUTURE_LEN:], copy=True))

                # record the time as well
                # if time_pt[PAST_LEN] != 0:
                #     past = np.zeros([PAST_LEN, 2])
                #     future = np.zeros([FUTURE_LEN, 2])
                #     past[:, 0] = time_pt[:PAST_LEN]
                #     past[:, 1] = bw_pt[:PAST_LEN]
                #     future[:, 0] = time_pt[-FUTURE_LEN:]
                #     future[:, 1] = bw_pt[-FUTURE_LEN:]
                #     x.append(past)
                #     y.append(future)

    x = np.vstack(x)
    y = np.vstack(y)
    
    return x, y
