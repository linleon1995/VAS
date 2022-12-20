import random
from pathlib import Path


def split(data_root, ratio=0.8):
    """AI is creating summary for split

    e.g., split(data_root='coffee_room_label')
    Args:
        data_root ([type]): [description]
        ratio (float, optional): [description]. Defaults to 0.8.
    """
    files = list(Path(data_root).glob('*.txt'))

    # XXX: seed
    random.shuffle(files)
    test_ratio = 1 - ratio
    train_num = int(len(files)*ratio)
    # test_num = int(len(files)*test_ratio)

    train_files, test_files = files[:train_num], files[train_num:]

    with open('train.txt', 'w+') as fw:
        for f in train_files:
            fw.write(f'{f.name}\n')

    with open('test.txt', 'w+') as fw:
        for f in test_files:
            fw.write(f'{f.name}\n')
