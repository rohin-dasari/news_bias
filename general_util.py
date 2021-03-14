import os

def get_logdir(base_dir, prefix=''):
    try:
        dirs = os.listdir(base_dir)
    except FileNotFoundError:
        dirs = []

    valid_dir = []
    for d in dirs:
        try:
            valid_dir.append(int(d.replace(prefix, '')))
        except ValueError:
            continue
    valid_dir_sorted = sorted(valid_dir)
    new_dir = valid_dir_sorted[-1]+1 if len(valid_dir_sorted) > 0 else 0
    return os.path.join(base_dir, prefix+str(new_dir))
