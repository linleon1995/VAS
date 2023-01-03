import copy
import json
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


class VAS_label_process():
    def __init__(self, data_dir, save_dir, save_mid_conversion=False):
        self.data_dir = Path(data_dir)
        self.save_dir = Path(save_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_mid_conversion = save_mid_conversion

    def load_json(self, filepath):
        f = open(filepath)
        json_data = json.load(f)
        return json_data

    def timestamp_to_framewise(self, timestamp):
        """This conversion is for Datumaro label format
        """
        framewise = copy.deepcopy(timestamp)

        current = framewise['items'][0]['annotations']
        for frame in framewise['items']:
            if frame['annotations']:
                if frame['annotations'] != current:
                    current = frame['annotations']
            else:
                frame['annotations'] = current
        return framewise

    def datumaro_to_actions(self, datumaro_label):
        """Convert Datumaro label to actions in text format"""
        labels = datumaro_label['categories']['label']['labels']
        actions = []

        for frame in datumaro_label['items']:
            annotations = frame['annotations']
            # assert len(annotations) == 1, 'annotation number error'
            if len(annotations) != 1:
                actions = None
                break

            label_idx = annotations[0]['label_id']
            actions.append(labels[label_idx]['name'])
        return actions

    def save_action_mapping(self, datumaro_label):
        with open(self.save_dir.joinpath(f'mapping.txt'), 'w+') as fw:
            categories = datumaro_label['categories']['label']['labels']
            # Let the backgound index be 0
            # XXX: bad imp
            categories = sorted(categories, key=lambda cat: cat['name'])
            for cat_idx, category in enumerate(categories):
                fw.write(f"{cat_idx} {category['name']}\n")

    def process_label_file(self, label_file):
        timestamp_label = self.load_json(label_file)

        # Convert timestamp label to framewise label
        framewise_labbel = self.timestamp_to_framewise(timestamp_label)
        if self.save_mid_conversion:
            framewise_dir = self.save_dir.joinpath('framewise')
            framewise_dir.mkdir(parents=True, exist_ok=True)
            json_object = json.dumps(framewise_labbel, indent=4)
            with open(str(framewise_dir.joinpath(label_file.name)), "w") as outfile:
                outfile.write(json_object)

        # Convert label format (Datumaro -> text)
        actions = self.datumaro_to_actions(framewise_labbel)
        actions_save_dir = self.save_dir.joinpath('groundTruth')
        actions_save_dir.mkdir(parents=True, exist_ok=True)
        if actions is not None:
            with open(actions_save_dir.joinpath(f'{label_file.stem}.txt'), 'w+') as fw:
                for action in actions:
                    fw.write(f"{action}\n")

    def run(self):
        for idx, label_file in enumerate(self.data_dir.glob('*.json')):
            print(f'Process label {idx} -> {str(label_file)}')
            label_file = Path(label_file)
            self.process_label_file(label_file)

        # mapping for index to actions
        datumaro_label_for_label_map = self.load_json(label_file)
        self.save_action_mapping(datumaro_label_for_label_map)

        #
        with open(self.save_dir.joinpath(f'filenames.txt'), 'w+') as fw:
            data = []
            for idx, label_file in enumerate(self.save_dir.joinpath('groundTruth').glob('*.txt')):
                data.append(f'{label_file.name}\n')
            data[-1] = data[-1][:-1]
            fw.writelines(data)


if __name__ == '__main__':
    # TODO: add split function
    data_dir = 'timestamp2'
    save_dir = 'frames2'

    vas_label_process = VAS_label_process(data_dir, save_dir, True)
    vas_label_process.run()
