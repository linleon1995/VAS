# TODO: argparse
# TODO: main function
from datetime import datetime
from multiprocessing import Process, Queue

import cv2

from scheduled_recording import check_time


def image_save(taskqueue, width, height, fps, frames_per_file):

    # 指定影片編碼
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #fourcc = cv2.VideoWriter_fourcc(*'H264')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    writer = None

    # def capture():

    #     return writer

    try:
        while True:
            check_time()

            image, frame_counter = taskqueue.get()

            # if image is None:
            #     break

            if frame_counter % frames_per_file == 0:

                if writer:
                    writer.release()

                # 建立 VideoWriter 物件（以數字編號）
                # index = int(frame_counter // frames_per_file)
                # writer = cv2.VideoWriter(f'output-{index}.mp4', fourcc, fps, (width, height))

                # 建立 VideoWriter 物件（以時間命名）
                now = datetime.now()
                timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
                writer = cv2.VideoWriter(
                    f'videos/output-{timestamp}.mp4', fourcc, fps, (width, height))

            # 儲存影像
            writer.write(image)
    except KeyboardInterrupt():
        print('KeyboardInterrupt sleeping')
        writer.release()


if __name__ == '__main__':

    # 開啟 RTSP 串流
    LAB_RTSP = 'rtsp://root:a1s2d3f4@192.168.50.161:554/live.sdp'
    vidCap = cv2.VideoCapture(LAB_RTSP)

    # 取得影像的尺寸大小
    width = int(vidCap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidCap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 取得影格率
    fps = vidCap.get(cv2.CAP_PROP_FPS)

    # 建立工作佇列
    taskqueue = Queue()

    # 計數器
    frame_counter = 0

    total_time = 100 * 3600  # 30 seconds per process
    file_time = 10 * 60  # 10 seconds per file

    # 總錄製幀數（30 秒鐘）
    total_frames = fps * total_time

    # 每個檔案的幀數（10 秒鐘）
    frames_per_file = fps * file_time

    # 建立並執行工作行程
    proc = Process(target=image_save, args=(
        taskqueue, width, height, fps, frames_per_file))
    proc.start()

    while frame_counter < total_frames:
        # 從 RTSP 串流讀取一張影像
        ret, image = vidCap.read()

        if ret:
            # 將影像放入工作佇列
            taskqueue.put((image, frame_counter))
            frame_counter += 1
        else:
            # 若沒有影像跳出迴圈
            break

    # 傳入 None 終止工作行程
    taskqueue.put((None, None))

    # 等待工作行程結束
    proc.join()

    # 釋放資源
    vidCap.release()
