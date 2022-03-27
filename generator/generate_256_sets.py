import os
import sys
sys.path.insert(0, os.path.abspath('.'))
from generator.generator import Generator
from generator.generator_params import GenParams
import numpy as np
import argparse
from pathlib import Path
from utils import ioutils

if __name__ == '__main__':
    classification_parser = ioutils.get_parser()
    parser = argparse.ArgumentParser('Script to test data generation')
    parser.add_argument('--root_dir', default='/mnt/media/gen_256_sets_40k')
    parser.add_argument('--port', type=int, default=1071)
    parser.add_argument('--load_path', type=str, default='/mnt/media')
    parser.add_argument('--num_imgs', type=int, default=40000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--idx', type=int, default=0) # index of generator
    parser.add_argument('--start_idx', type=int, default=None)
    parser.add_argument('--end_idx', type=int, default=None)
    parser.add_argument('--num_nodes', type=int, default=5)
    args = parser.parse_args()

    gen = Generator(dataset_dir=Path(args.root_dir).expanduser(), port=args.port,
                    load_path=args.load_path, skybox_preload=True)
    ranges = GenParams.get_ranges()

    RNG = np.random.RandomState(args.seed)
    params = [
        'pose_rot', 'pose_scale', 'lighting_intensity', 'lighting_color', 'lighting_dir','blur', 'backgr', 'materials']
    num_datasets = 256

    start_idx = args.start_idx or args.idx*np.ceil(num_datasets/args.num_nodes)
    end_idx = args.end_idx or min(num_datasets, (args.idx+1)*np.ceil(num_datasets/args.num_nodes))
    for i in range(int(start_idx), int(end_idx)):
        gp = GenParams()
        param_vals = '{:08b}'.format(i) # get the binary idx
        for param, val in zip(params, param_vals):
            gp[param] = int(val)
        gp.materials = 2*gp.materials + 1
        img_folder = gen.gen_data(gp, num_imgs=args.num_imgs)
        print('Dataset generated in directory : {}'.format(img_folder))
