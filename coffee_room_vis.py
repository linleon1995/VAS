from utils.vas_vis_utils import vas_videos


def main():
    dataset = 'coffee_room'

    gt_root = r'C:\Users\test\Desktop\Leon\Projects\MS_TCN2\data\coffee_room\groundTruth'
    recog_root = r'C:\Users\test\Desktop\Leon\Projects\UVAST\exp\coffee_room2_add\model\coffee_room\split_2\AD\recong'
    video_root = r'C:\Users\test\Desktop\Leon\Datasets\coffee_room'
    data_root = r'C:\Users\test\Desktop\Leon\Projects\MS_TCN2\data'

    gt_root = r'C:\Users\test\Desktop\Leon\Projects\MS_TCN2\data\coffee_room\groundTruth'
    recog_root = r'C:\Users\test\Desktop\Leon\Projects\MS_TCN2\results\coffee_room\split_4'
    video_root = r'C:\Users\test\Desktop\Leon\Projects\VAS\exp\New folder'
    video_root = r'C:\Users\test\Desktop\Leon\Datasets\coffee_room'
    data_root = r'C:\Users\test\Desktop\Leon\Projects\MS_TCN2\data'

    # gt_root = r'C:\Users\test\Desktop\Leon\Projects\MS_TCN2\data\coffee_room\groundTruth'
    # recog_root = r'C:\Users\test\Desktop\Leon\Projects\UVAST\coffee_room2\inference2'
    # recog_root = r'C:\Users\test\Desktop\Leon\Projects\UVAST\exp\coffee_room2_add\results\coffee_room\split_2\AD\recong'
    # video_root = r'C:\Users\test\Desktop\Leon\Datasets\coffee_room\events_door'
    # data_root = r'C:\Users\test\Desktop\Leon\Projects\MS_TCN2\data'

    vas_videos(dataset, data_root, video_root,
               gt_root, recog_root, save_root='exp/x3d_m_add52_test_mstcn')


if __name__ == '__main__':
    # f = r'C:\Users\test\Desktop\Leon\Datasets\coffee_room\events_door_feature\x3d_m\20221102150450_coffee_video_people2.npy'
    # import numpy as np
    # print(np.load(f).shape)
    main()
