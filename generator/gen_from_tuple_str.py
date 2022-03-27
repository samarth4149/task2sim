import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import argparse
from utils import ioutils
from pathlib import Path
from generator.generator import Generator
from generator.generator_params import GenParams

if __name__ == '__main__':
    classification_parser = ioutils.get_parser()
    parser = argparse.ArgumentParser('Script to generate images for full set from tuple str')
    parser.add_argument('--root_dir', default='/mnt/media')
    parser.add_argument('--port', type=int, default=1071)
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--num_imgs', type=int, default=1300000)
    parser.add_argument('--simple_objs', action='store_true', default=False)
    parser.add_argument('--tuple_str', type=str, default='')
    args = parser.parse_args()

    gen = Generator(dataset_dir=Path(args.root_dir).expanduser(), port=args.port,
                    load_path=args.load_path, skybox_preload=False, simple_objs=args.simple_objs)

    image_folder = gen.gen_data(GenParams.from_tuple_str(args.tuple_str), num_imgs=args.num_imgs)
    print('Dataset : {} generated in {}'.format(args.tuple_str, image_folder))