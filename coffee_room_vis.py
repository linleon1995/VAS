from video_display import vas_videos


def main():
    dataset = 'coffee_room'
    gt_root = r'C:\Users\test\Desktop\Leon\Projects\MS-TCN2\data\coffee_room\groundTruth'
    recog_root = r'C:\Users\test\Desktop\Leon\Projects\UVAST\coffee_room2\inference'
    video_root = r'C:\Users\test\Desktop\Leon\Datasets\coffee_room_door_event_dataset'
    data_root = r'C:\Users\test\Desktop\Leon\Projects\MS-TCN2\data'
    vas_videos(dataset, data_root, video_root,
               gt_root, recog_root, save_root='results')


if __name__ == '__main__':
    main()
