"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.S3DISDataLoader import ScannetDatasetWholeScene
from data_utils.SonarDataLoader import ShapeNetDataset,ShapeNetDatasetWholeScene
from data_utils.indoor3d_util import g_label2color
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import provider
import numpy as np
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

#classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
#           'board', 'clutter']

classes = ['Box','Man','Other','Water_surface']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in testing [default: 16]')
    parser.add_argument('--gpu', type=str, default='1', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=2048, help='point number [default: 4096]')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--visual', action='store_true', default=False, help='visualize result [default: False]')
    #parser.add_argument('--test_area', type=int, default=5, help='area for testing, option: 1-6 [default: 5]')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting [default: 5]')
    return parser.parse_args()


def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    experiment_dir = 'log/sem_seg/' + args.log_dir
    visual_dir = experiment_dir + '/visual/'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    NUM_CLASSES = 4
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point

    #root = 'data/s3dis/stanford_indoor3d/'
    root = '/home/data6T/pxy/pointnet.pytorch/sonar_data/'

    #TEST_DATASET_WHOLE_SCENE = ScannetDatasetWholeScene(root, split='test', test_area=args.test_area, block_points=NUM_POINT)
    # TEST_DATASET_WHOLE_SCENE = ShapeNetDatasetWholeScene(root, split='test', block_points=NUM_POINT)
    # log_string("The number of test data is: %d" % len(TEST_DATASET_WHOLE_SCENE))

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    classifier = classifier.eval()
    x = torch.randn(24, 3, 2048).cuda()
    # print(classifier)
    # exit(0)

    import time
    with torch.no_grad():
        classifier = classifier.eval()
        times_list = []
        for i in range(100):
            st = time.time()
            y = classifier(x)
            ed = time.time()
            times_list.append(ed-st)
        print(f'time {sum(times_list) / len(times_list)}')

if __name__ == '__main__':
    args = parse_args()
    main(args)
