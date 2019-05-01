import scipy.io
import numpy as np
from eval_tools.bbox_overlap import bbox_overlap


def evaluation(norm_pred_list, gt_dir, name, legend_name):
    # Load
    infos = scipy.io.loadmat(gt_dir)
    file_list = np.array(infos['file_list']).squeeze()
    face_bbx_list = np.array(infos['face_bbx_list']).squeeze()
    gt_list = np.array(infos['gt_list']).squeeze()

    # Declare
    IoU_thresh = 0.5
    event_num = 61
    thresh_num = 1000
    count_face = 0
    recall, precision = 0.0, 0.0
    print('-----')
    for i in range(event_num):
        img_list = file_list[i][0]
        gt_bbox_list = face_bbx_list[i][0]
        pred_list = norm_pred_list[i]
        sub_gt_list = gt_list[i][0]
        for j in range(len(img_list)):
            gt_bbox = np.array(gt_bbox_list[j])
            pred_info = np.array(pred_list[j])
            keep_index = np.array(sub_gt_list[j])
            count_face += len(keep_index)
            if len(gt_bbox) == 0 or len(pred_info) == 0:
                continue

            ignore = np.zeros(len(gt_bbox))

            if not len(keep_index):
                # 猜测是sub set的编号
                ignore[keep_index] = 1
            pred_recall, proposal_list = image_evaluation(pred_info, gt_bbox, ignore, IoU_thresh)
            img_pr_info = image_pr_info(thresh_num, pred_info, proposal_list, pred_recall)

            precision += img_pr_info[0]
            recall += img_pr_info[1]

    precision = (precision / recall) if recall != 0 else 0
    recall = (recall / count_face) if count_face != 0 else 0

    f1_score = (2 * precision * recall) / (precision + recall)
    print(f'{legend_name}: {name}\'s f1_score is {100*f1_score:.3f}')


def image_evaluation(pred_info, gt_bbx, ignore, IoU_thresh):
    # 预测的
    pred_recall = np.zeros((pred_info.shape[0],), dtype=np.int8)
    # 真实的
    recall_list = np.zeros((gt_bbx.shape[0],), dtype=np.int8)
    # proposal是什么？
    proposal_list = np.ones((pred_info.shape[0],), dtype=np.int8)
    #
    pred_info[:, 2] += pred_info[:, 0]
    pred_info[:, 3] += pred_info[:, 1]
    gt_bbx[:, 2] += gt_bbx[:, 0]
    gt_bbx[:, 3] += gt_bbx[:, 1]

    for h in range(len(pred_info)):
        overlap_list = bbox_overlap(gt_bbx, pred_info[h])
        idx = np.argmax(overlap_list)
        max_overlap = overlap_list[idx]
        if max_overlap >= IoU_thresh:
            if ignore[idx] == 0:
                recall_list[idx] = -1
                proposal_list[h] = -1
            # ignore=1的idx才是我们需要的，所以这里如果再满足这个是=0，就是我们要的
            elif recall_list[idx] == 0:
                recall_list[idx] = 1
        # 累加：看到这个位置他的recall是多少
        r_keep_index = np.where(recall_list == 1)
        pred_recall[h] = len(r_keep_index)
    return pred_recall, proposal_list


def image_pr_info( proposal_list, pred_recall):
    p_index = np.where(proposal_list == 1)
    precision = len(p_index)
    recall = pred_recall[-1]
    return [precision, recall]
