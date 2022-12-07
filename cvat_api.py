# import
import requests
from io import BytesIO
import zipfile
import json
# def


def login(url, username, password):
    login_json = {"username": username, "password": password}
    login_resp = requests.post(url='/'.join([url, 'api/auth/login']),
                               json=login_json)
    user_cookie = login_resp.cookies
    return user_cookie


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


if __name__ == '__main__':
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
