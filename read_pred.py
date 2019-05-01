import scipy.io
import numpy as np


def read_pred(file_dir, gt_dir):
    infos = scipy.io.loadmat(gt_dir)
    pred_list = []
    file_list = np.array(infos['file_list']).squeeze()
    event_list = np.array(infos['event_list']).squeeze()

    for i in range(61):
        img_list = file_list[i]
        # 格式不匹配
        img_num = len(img_list)
        bbox_list = []

        for j in range(img_num):
            with open(f'{file_dir}/{event_list[i][0]}/{img_list[j][0][0]}.txt', 'r') as f:
                tmp = f.readlines()
            if not tmp:
                bbox_list.append([])
                continue
            # 第一行是路径
            # 第二行是bbox的数量
            bbox_num = len(tmp) - 2
            bbox = np.zeros((bbox_num, 4))
            if bbox_num == 0:
                bbox_list.append([])
                continue
            else:
                for k, (line) in enumerate(tmp[2:]):
                    _raw = line[:-1].split()
                    bbox[k] = [float(_raw[0]), float(_raw[1]), float(_raw[2]), float(_raw[3])]
            bbox_list.append(bbox)
        pred_list.append(bbox_list)

    return pred_list
