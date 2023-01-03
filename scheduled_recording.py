import os
import datetime
from time import sleep

import cv2


def get_capture_attribute(capture):
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return fps, width, height


def record(schedule_start_time, schedule_end_time, location, url):
    check_time(schedule_start_time, schedule_end_time)
    os.system(f'python video_capture.py --loc={location}')


def check_time(schedule_start_time=None, schedule_end_time=None):
    # set default time
    if not schedule_start_time:
        schedule_start_time = datetime.time(hour=8, minute=0, second=0)

    if not schedule_end_time:
        schedule_end_time = datetime.time(hour=20, minute=0, second=0)

    # check time
    current_datetime, current_time = get_current_time()
    time_in_range = is_time_in_range(start=schedule_start_time,
                                     end=schedule_end_time,
                                     current_time=current_time)

    # check working day & time range
    if not time_in_range or current_datetime.isoweekday() in [6, 7]:
        sleep_until_next_record_time(current_datetime, schedule_start_time)
        # assert 0


def sleep_until_next_record_time(current_datetime, schedule_start_time):
    week_day = current_datetime.isoweekday()
    if week_day in [6, 7]:
        day_plus = 7 - week_day + 1
    else:
        if current_datetime.hour < schedule_start_time.hour:
            day_plus = 0
        else:
            day_plus = 1

    wake_time = current_datetime
    wake_time = wake_time.replace(
        day=wake_time.day+day_plus,
        hour=schedule_start_time.hour,
        minute=schedule_start_time.minute,
        second=schedule_start_time.second
    )
    print(
        f'current time {current_datetime} not in range, so sleep until {wake_time}')
    sleep_until(wake_time)


def sleep_until(wake_time):
    current_datetime, current_time = get_current_time()
    if wake_time > current_datetime:
        sleep_time = wake_time - current_datetime
    else:
        sleep_time = 0
    sleep_second = sleep_time.days*24*3600 + sleep_time.seconds
    sleep(sleep_second)


def get_current_time():
    current_datetime = datetime.datetime.now()
    current_time = datetime.time(hour=current_datetime.hour,
                                 minute=current_datetime.minute,
                                 second=current_datetime.second)
    return current_datetime, current_time


def is_time_in_range(start, end, current_time):
    if start <= end:
        return start <= current_time <= end
    else:
        return start <= current_time or current_time <= end


if __name__ == '__main__':
    # parameters
    schedule_start_time = datetime.time(hour=8, minute=0, second=0)
    schedule_end_time = datetime.time(hour=20, minute=0, second=0)
    location = 'coffee'
    url_map = {
        'coffee': 'rtsp://ditsol:1234567@192.168.1.141:554/stream1',
        'lab': 'rtsp://root:a1s2d3f4@192.168.50.161:554/live.sdp'
    }
    url = url_map[location]

    # run
    filename = None
    idx = 0
    while True:
        if idx > 10:
            break
        idx += 1
        try:
            filename = record(schedule_start_time,
                              schedule_end_time, location, url)
        except:
            if filename:
                print(f'{filename} record failed')
