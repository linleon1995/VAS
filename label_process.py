import json
import copy
from pathlib import Path


def timestamp_to_framewise(label_file):
    f = open(label_file)
    timestamp = json.load(f)
    framewise = copy.deepcopy(timestamp)

    current = framewise['items'][0]['annotations']
    for frame in framewise['items']:
        if frame['annotations']:
            if frame['annotations'] != current:
                current = frame['annotations']
        else:
            frame['annotations'] = current
    return framewise


if __name__ == '__main__':
    data_dir = Path('timestamp')
    save_dir = Path('frames')
    save_dir.mkdir(parents=True, exist_ok=True)

    for label_file in data_dir.glob('*.json'):
        framewise = timestamp_to_framewise(label_file)
        json_object = json.dumps(framewise, indent=4)
        with open(str(save_dir.joinpath(label_file.name)), "w") as outfile:
            outfile.write(json_object)
