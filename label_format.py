import json
from pathlib import Path


def format_transform(datumaro_label):
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


if __name__ == '__main__':
    save_dir = Path('coffee_room_label')
    save_dir.mkdir(parents=True, exist_ok=True)

    for idx, label_file in enumerate(Path('frames').glob('*.json')):
        datumaro_label = json.load(f := open(label_file))
        if idx == 0:
            with open(save_dir.joinpath(f'mapping.txt'), 'w+') as fw:
                categories = datumaro_label['categories']['label']['labels']
                # Let the backgound index be 0
                categories = sorted(categories, key=lambda cat: cat['name'])
                for cat_idx, category in enumerate(categories):
                    fw.write(f"{cat_idx} {category['name']}\n")

        actions = format_transform(datumaro_label)
        if actions is not None:
            with open(save_dir.joinpath(f'{label_file.stem}.txt'), 'w+') as fw:
                for action in actions:
                    fw.write(f"{action}\n")

    pass
