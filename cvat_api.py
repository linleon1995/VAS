import requests
from io import BytesIO
import zipfile
import json
from pathlib import Path


def login(url, username, password):
    login_json = {"username": username, "password": password}
    login_resp = requests.post(url='/'.join([url, 'api/auth/login']),
                               json=login_json)
    user_cookie = login_resp.cookies
    return user_cookie


def get_label_config():
    labels = [
        {
            "name": "open_door",
            "id": 13,
            "color": "#ff0000",
            "type": "any",
            "attributes": []
        },
        {
            "name": "close_door",
            "id": 14,
            "color": "#7f0000",
            "type": "any",
            "attributes": []
        },
        {
            "name": "stick_around",
            "id": 15,
            "color": "#00ff00",
            "type": "any",
            "attributes": []
        },
        {
            "name": "turn_on_the_light",
            "id": 16,
            "color": "#0000ff",
            "type": "any",
            "attributes": []
        },
        {
            "name": "turn_off_the_light",
            "id": 17,
            "color": "#00007f",
            "type": "any",
            "attributes": []
        },
        {
            "name": "background",
            "id": 18,
            "color": "#000000",
            "type": "any",
            "attributes": []
        }
    ]
    return labels


def create_project(url, auth, name: str, labels: dict) -> None:
    params = {
        "name": name,
        "labels": labels,
    }
    response = requests.post(url='/'.join([url, 'api/projects']),
                             #  cookies=cookies,
                             json=params,
                             auth=auth
                             )
    return response


def create_task(url, auth, name: str) -> None:
    params = {
        "name": name,
        "project_id": 23
    }
    # task_data = {
    #     "image_quality": 70,
    #     "start_frame": 0,
    #     "stop_frame": 450,
    #     "client_files": [r'C:\Users\test\Desktop\Leon\Projects\VAS\20221102090511_coffee_video_88997_89223.mp4'],
    #     "storage_method": "file_system",
    #     "storage": "local",
    # }
    # resources = ['bird.jpg']
    # files = {f'client_files[{i}]': open(f, 'rb')
    #          for i, f in enumerate(resources)}

    response = requests.post(url='/'.join([url, 'api/tasks']),
                             auth=auth,
                             json=params)
    return response


def attach_data(url, auth, task_id, vid_file, image_quality: int = 70):
    with open(vid_file, 'rb') as vid_obj:
        data_server_files = {
            # "image_quality": 70,
            # 'client_files[0]': [r'http://192.168.1.146:8000/20221102090511_coffee_video_88997_89223.mp4'],
            'client_files[0]': vid_obj
            # 'remote_files[0]': [r'https://i.natgeofe.com/n/d472dd3c-8d38-4eed-ae62-7472a12a17de/secretary-bird-thumbnail-nationalgeographic_2331336_3x2.jpg'],
            # 'server_files[0]': [r'http://192.168.1.146:8000/20221102090511_coffee_video_88997_89223.mp4'],
            # 'remote_files[0]': [r'https://assets.mixkit.co/videos/preview/mixkit-going-down-a-curved-highway-through-a-mountain-range-41576-large.mp4'],
            # "server_files[0]": [r'20221102090511_coffee_video_88997_89223.mp4']
        }
        response = requests.post(url='/'.join([url, f'api/tasks/{task_id}/data']),
                                 verify=False,
                                 auth=auth,
                                 data={"image_quality": image_quality},
                                 files=data_server_files
                                 #  data=data_server_files
                                 )
    return response


def get_tasks(url, cookies, page_size):
    params = {'page_size': page_size}
    data = requests.get(url='/'.join([url, 'api/tasks']),
                        cookies=cookies,
                        params=params)
    data = data.json()
    return data


def download_annotations(url, cookies, task_id, annotation_format):
    downloader = create_downloader(url, cookies, task_id, annotation_format)
    zip_file = zipfile.ZipFile(BytesIO(downloader.content))
    filename = zip_file.namelist()[0]
    content_byte = zip_file.read(filename)
    annotations = json.loads(content_byte.decode('utf8'))
    return annotations


def create_downloader(url, cookies, task_id, annotation_format):
    params = {'action': 'download', 'format': annotation_format}
    downloader = requests.get(url='/'.join([url,
                                            f'api/tasks/{task_id}/annotations/']),
                              cookies=cookies,
                              params=params,
                              stream=True)
    while downloader.status_code != 200:
        downloader = create_downloader(url, cookies, task_id,
                                       annotation_format)
    return downloader


def dowmload_annotation():
    # parameters
    url = 'http://192.168.50.89:8080'
    username = 'admin'
    password = 'a1s2d3f4'
    annotation_format = 'Datumaro 1.0'
    page_size = 100

    # login
    user_cookie = login(url, username, password)

    # get tasks
    tasks = get_tasks(url, user_cookie, page_size)

    # download annotations
    for data in tasks['results']:
        task_id = data['id']
        task_name = data['name']
        state = data['segments'][0]['jobs'][0]['state']
        if state == 'completed' or state == 'rejected':
            annotations = download_annotations(url, user_cookie, task_id,
                                               annotation_format)
            filename = task_name.split('_', 2)[-1]
            with open(f'{filename}.json', 'w') as file:
                json.dump(annotations, file)


def upload_annots_and_images():
    # parameters
    url = 'http://192.168.50.89:8080'
    username = 'admin'
    password = 'a1s2d3f4'
    annotation_format = 'Datumaro 1.0'
    page_size = 100
    vid_root = r'C:\Users\test\Desktop\Leon\Datasets\coffee_room\events_door'
    vid_files = Path(vid_root).glob('*.mp4')

    # login
    user_cookie = login(url, username, password)
    auth = (username, password)

    # create project
    labels = get_label_config()
    create_project(url, auth, name='CRDE_new_collection', labels=labels)

    # create tasks
    start_task_id = 94
    for task_id, vid_file in enumerate(vid_files, start_task_id):
        create_task(url, auth, name=vid_file.stem)
        attach_data(url, auth, task_id=task_id, vid_file=str(vid_file))
    # upload annotations

    # # get tasks
    # tasks = get_tasks(url, user_cookie, page_size)

    # # download annotations
    # for data in tasks['results']:
    #     task_id = data['id']
    #     task_name = data['name']
    #     state = data['segments'][0]['jobs'][0]['state']
    #     if state == 'completed' or state == 'rejected':
    #         annotations = download_annotations(url, user_cookie, task_id,
    #                                            annotation_format)
    #         filename = task_name.split('_', 2)[-1]
    #         with open(f'{filename}.json', 'w') as file:
    #             json.dump(annotations, file)


if __name__ == '__main__':
    # dowmload_annotation()
    upload_annots_and_images()
