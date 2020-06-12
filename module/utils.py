import os
import numpy as np
import random
import torch

def clear_sub_files(pathes):
    for path in pathes:
        file_or_dir = os.listdir(path)
        if len(file_or_dir) > 0:
            for name in file_or_dir:
                file_path = os.path.join(path, name)
                print("remove old file:", file_path)
                os.remove(file_path)


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def add_json_to_class(jdata,cdata):
    for key, val in jdata.items():
        if  val:
            setattr(cdata, key,val)
    return cdata

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def sec_to_h(secs):
    h = int(secs/3600)
    m = int((secs -h*3600)/60)
    s = int(secs -3600*h-60*m)
    return [h,m,s]