import os
import random

def gen_file_list(file_path):
    return sorted(list(os.listdir(file_path)))


def list_files(directory):
    r = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            if (len(files) > 3) & name.endswith('.dcm'):
                r.append(os.path.join(root, name))
    return r


def random_shuffle(*lists):
    idx = list(zip(*lists))
    random.shuffle(idx)
    return zip(*idx)


def rescale(value, num_files):
    new_value = 100 * (value / num_files)
    return round(new_value)

def clear_temp():
    for root, dirs, files in os.walk('./temp'):
        for name in files:
            os.remove('./temp/' + name)
