from argparse import ArgumentParser
import getpass
import requests
from pprint import pprint
import json
import logging
import http.client as http_client
import time

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning) # to suppress warnings that "Unverified HTTPS request is being made"

def set_http_verbose():
    http_client.HTTPConnection.debuglevel = 1

    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    requests_log = logging.getLogger("requests.packages.urllib3")
    requests_log.setLevel(logging.DEBUG)
    requests_log.propagate = True

def print_hrule(name = ""):
    if name:
        print("=" * 5 + " " + str(name) + " " + "=" * 80)
    else:
        print("=" * 80)

def get_user_password_from_file(password_file):
    with open(password_file) as f:
        for line in f:
            line = line.strip()
            if ':' not in line or line.startswith('#'):
                continue
            chunks = line.split(':')
            assert len(chunks) == 2, "Wrong password_file {}".format(password_file)
            return chunks
    raise RuntimeError("Wrong password file")

def get_user_password_from_input():
    user = getpass.getuser()
    answer = input("User = {}? [Y/n]: ".format(user))
    if answer and not answer.lower().startswith('y'):
        user = input('User: ')

    password = getpass.getpass()
    return user, password

def create_task(full_relative_path, bug_tracker, cur_user_info, user, password):
    res = {"status": False, "full_relative_path": full_relative_path}

    data_task_create = {
        "name": full_relative_path,
        "owner": cur_user_info["id"],
        "image_quality": 75,
        "bug_tracker": bug_tracker,
        "labels": [ #'obj @text=class:""'
                {
                    "name": "obj",
                    "attributes": [
                        {
                            "name": "class",
                            "mutable": False,
                            "input_type": "text",
                            "default_value": "",
                            "values": [""],
                        },
                    ]
                },
            ]
    }

    print("")
    print_hrule("BEGIN: Create task")
    task_creation_resp = requests.post('https://cvat-icv.inn.intel.com/api/v1/tasks', verify=False, auth=(user, password), json=data_task_create)
    print_hrule("END: Create task")

    print("task_creation_resp.status_code =", task_creation_resp.status_code)
    print("task_creation_resp.json =")
    pprint(task_creation_resp.json())
    if task_creation_resp.status_code != 201:
        print("CANNOT CREATE TASK")
        return res
    task_id = task_creation_resp.json()["id"]
    res["task_id"] = task_id

    data_server_files = {
#            'client_files': [],
#            'remote_files': [],
            "server_files[0]": [full_relative_path]
    }

    print("")
    print_hrule("BEGIN: Point video to task")
    server_files_resp = requests.post('https://cvat-icv.inn.intel.com/api/v1/tasks/{}/data'.format(task_id), verify=False, auth=(user, password), data=data_server_files)
    print_hrule("END: Point video to task")

    print("server_files_resp.status_code =", server_files_resp.status_code)
    print("server_files_resp.json =")
    pprint(server_files_resp.json())
    if int(server_files_resp.status_code) not in (201, 202):
        print("CANNOT SET SERVER FILES")
        return res

    print("Task for full_relative_path='{}' is added".format(full_relative_path))

    status_resp_json = {}
    while True:
        print_hrule("BEGIN: Status")
        status_files_resp = requests.get('https://cvat-icv.inn.intel.com/api/v1/tasks/{}/status'.format(task_id), verify=False, auth=(user, password))
        print_hrule("END: Status")

        print("status_files_resp.status_code =", status_files_resp.status_code)
        if status_files_resp.status_code != 200:
            print("CANNOT GET STATUS")
            return res
        status_resp_json = status_files_resp.json()
        print("status_files_resp.json =")
        pprint(status_resp_json)
        if status_resp_json.get('state', "") in ("Finished", "Failed"):
            break

        time.sleep(1)

    if status_resp_json.get('state', "") == "Finished":
        print("Task is created and video is decoded for full_relative_path = '{}'".format(full_relative_path))
    else:
        print("ERROR DURING CREATION OF THE TASK '{}'".format(full_relative_path))
        return res

    #task_id = 2536
    print_hrule("BEGIN: Get Job Id")
    job_id_resp = requests.get('https://cvat-icv.inn.intel.com/api/v1/tasks/{}'.format(task_id), verify=False, auth=(user, password))
    print_hrule("END: Get Job Id")
    if job_id_resp.status_code != 200:
        print("CANNOT GET JOB ID, status code =", job_id_resp.status_code)
        pprint(job_id_resp.json())
        return res
    job_id_json = job_id_resp.json()
    pprint(job_id_json)
    assert "segments" in job_id_json
    segments = list(job_id_json["segments"])
    assert segments
    assert len(segments) == 1
    assert "jobs" in segments[0]
    jobs = segments[0]["jobs"]
    assert len(jobs) == 1
    job_id = jobs[0]["id"]
    url_for_job = "https://cvat-icv.inn.intel.com/?id={}".format(job_id)
    print("url_for_job =", url_for_job)
    res["url_for_job"] = url_for_job
    res["status"] = True
    return res

def print_table_for_jira(list_all_done_res):
    for res in list_all_done_res:
        assert res["status"]
        print("|{}|[{}]|".format(res["full_relative_path"], res["url_for_job"]))

def main():
    parser = ArgumentParser()
    parser.add_argument('--password_file', help='Path to a password file: text file with one line in format <username>:<password>')
    parser.add_argument('--path', help='Path to a video to create task, relative to the textile folder, so textile/<path> will be the name of the task')
    parser.add_argument('--filelist', help='Path to a text file each line of which is a path to a video to create task (see --path command line parameter above)')
    parser.add_argument('--bug_tracker', required=True, help='URI of JIRA bug')
    parser.add_argument("-v", "--verbose", action="store_true", help="If HTTP requests should be logged")

    args = parser.parse_args()

    if args.path and args.filelist:
        print("Only one command line argument '--path' or '--filelist' is allowed")
        return
    if not (args.path or args.filelist):
        print("One command line argument '--path' or '--filelist' is required")
        return

    if args.verbose:
        set_http_verbose()

    if args.password_file:
        user, password = get_user_password_from_file(args.password_file)
    else:
        user, password = get_user_password_from_input()

    #path = "20190509/mingyin_raw_videos/video_0418/6/8 00_16_49-00_19_24.mp4"

    print_hrule("BEGIN: User info")
    resp = requests.get('https://cvat-icv.inn.intel.com/api/v1/users/self', verify=False, auth=(user, password))
    print_hrule("END: User info")

    if resp.status_code != 200:
        print("Wrong username/password")
        return False

    cur_user_info = resp.json()
    print("Get user info:")
    pprint(cur_user_info)

    if args.path:
        path = args.path
        full_relative_path = "textile/" + path
        print("full_relative_path = '{}'".format(full_relative_path))

        res = create_task(full_relative_path, args.bug_tracker, cur_user_info, user, password)
        if res["status"]:
            print("DONE")
        else:
            print("FAILED")
        return

    list_all_done_res = []
    list_failed_videos = []
    with open(args.filelist) as f_filelist:
        for path in f_filelist:
            path = path.strip()
            assert not path.startswith(".")

            full_relative_path = "textile/" + path
            print("full_relative_path = '{}'".format(full_relative_path))

            res = create_task(full_relative_path, args.bug_tracker, cur_user_info, user, password)
            print("")
            if res["status"]:
                print("DONE")
                list_all_done_res.append(res)
            else:
                print("FAILED {}!!!".format(full_relative_path))
                list_failed_videos.append(full_relative_path)
                #break
            print("")
            print_table_for_jira(list_all_done_res)
            print("")
            print("")
    print_table_for_jira(list_all_done_res)
    print("ALL DONE")
    print("")

    print("Failed videos:")
    for p in list_failed_videos:
        print("    ", p)



if __name__ == '__main__':
    main()
