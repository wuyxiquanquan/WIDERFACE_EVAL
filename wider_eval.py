from eval_tools.read_pred import read_pred
from eval_tools.evaluation import evaluation

pred_dir = '/home/snowcloud/widerface/finallll'
gt_dir = '/home/snowcloud/widerface/eval_tools/ground_truth/wider_face_val.mat'

# 读取pred的所有bbox
# 目录结构是
pred_list = read_pred(pred_dir, gt_dir)

# 三个子集
setting_name_list = ['easy_val', 'medium_val', 'hard_val']

#
method_name = 'segmentation is all you need'

for name in setting_name_list:
    print(f'Current evaluation setting {name}\n')
    gt_dir = f'/home/snowcloud/widerface/eval_tools/ground_truth/wider_{name}.mat'
    evaluation(pred_list, gt_dir, name, method_name)
