import requests
from io import BytesIO
import zipfile
import json
from pathlib import Path
from typing import Tuple


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


def create_task(url: str, auth: Tuple, name: str, project_id: int) -> None:
    params = {
        "name": name,
        "project_id": project_id
    }

    response = requests.post(url='/'.join([url, 'api/tasks']),
                             auth=auth,
                             json=params)
    return response


def attach_data(url, auth, task_id, vid_file, image_quality: int = 70):
    with open(vid_file, 'rb') as vid_obj:
        data_server_files = {
            'client_files[0]': vid_obj
        }
        response = requests.post(url='/'.join([url, f'api/tasks/{task_id}/data']),
                                 verify=False,
                                 auth=auth,
                                 data={"image_quality": image_quality},
                                 files=data_server_files
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


def dowmload_annotation(save_dir=None, task_ids=None):
    # parameters
    url = 'http://192.168.50.89:8080'
    username = 'admin'
    password = 'a1s2d3f4'
    annotation_format = 'Datumaro 1.0'
    page_size = 500
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # login
    user_cookie = login(url, username, password)

    # get tasks
    tasks = get_tasks(url, user_cookie, page_size)

    # download annotations
    for data in tasks['results']:
        task_id = data['id']
        task_name = data['name']
        # TODO:
        # if task_id < 94 or task_id > 387:
        # continue
        if task_id not in task_ids:
            continue

        if data['segments']:
            state = data['segments'][0]['jobs'][0]['state']
            stage = data['segments'][0]['jobs'][0]['stage']
            if state == 'completed' or state == 'rejected' or stage == 'acceptance':
                annotations = download_annotations(url, user_cookie, task_id,
                                                   annotation_format)
                filename = task_name
                # filename = task_name.split('_', 2)[-1]
                print(f'Download label {task_id} -> {filename}')
                if save_dir is not None:
                    filename = str(save_dir.joinpath(filename))
                with open(f'{filename}.json', 'w') as file:
                    json.dump(annotations, file)


def upload_annots_and_images():
    # TODO: Warping the function
    # parameters
    url = 'http://192.168.50.89:8080'
    username = 'admin'
    password = 'a1s2d3f4'
    annotation_format = 'Datumaro 1.0'
    page_size = 100
    vid_root = r'C:\Users\test\Desktop\Leon\Datasets\coffee_room\events_door'
    vid_files = Path(vid_root).glob('*.mp4')

    # login
    # cookies is not working on POST API, which currently not sure the reason
    # user_cookie = login(url, username, password)
    auth = (username, password)

    # create project
    labels = get_label_config()
    response = create_project(
        url, auth, name='CRDE_new_collection', labels=labels)
    response_content = response.json()
    project_id = response_content['id']

    # create tasks
    # TODO: decide start_task_id automatically
    start_task_id = 94
    for task_id, vid_file in enumerate(vid_files, start_task_id):
        create_task(url, auth, name=vid_file.stem, project_id=project_id)
        attach_data(url, auth, task_id=task_id, vid_file=str(vid_file))


if __name__ == '__main__':
    save_dir = r'timestamp2'
    task_ids = list(range(94, 388))
    dowmload_annotation(save_dir=save_dir, task_ids=task_ids)
    # upload_annots_and_images()
