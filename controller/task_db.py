import os
import sys
sys.path.insert(0, os.path.abspath('.'))
from t2v.task2vec import Task2Vec, Embedding
from t2v.models import get_model
from data.prep_datasets import prep_datasets
from classifier.config import ClassifierConfig
from argparse import Namespace
import torch
from argparse import ArgumentParser


def get_task_dim(task_db:dict):
    # Pull out an element from the db (and embedding) and get dimension
    return task_db[next(iter(task_db))].hessian.shape[0]

def load_task_db(task_db_paths):
    task_db = {}
    for path in task_db_paths:
        # Just load the task2vec vectors of all datasets in the database from a file
        loaded_db = torch.load(path)
        for key in loaded_db:
            task_db[key] = Embedding(loaded_db[key]['hessian'], loaded_db[key]['scale'])
    return task_db

def compute_task2vecs(args : Namespace):
    # compute the task2vec for the train splits of all datasets and save in a file
    dataset_list = [
        'CropDisease',
        'EuroSAT',
        'SVHN',
        'ChestX',
        'Sketch',
        'DTD',
        'Flowers102',
        'DeepWeeds',
        'Resisc45',
        'Omniglot',
        'ISIC',
        'Kaokore',

        'CUB',
        'PacsC',
        'PacsS',
        'AID',
        'USPS',
        'FMD',
        'CactusAerial',
        'ChestXPneumonia'
    ]

    cfg = ClassifierConfig()
    cfg.defrost()
    cfg.DOWNSTREAM_EVAL = 'task2vec' # So train transform = val transform
    cfg.freeze()
    args.quick=False

    t2v_task_db = {}
    for d in dataset_list:
        cfg.defrost()
        cfg.DATASET = d
        cfg.RESUME = args.backbone_path
        cfg.freeze()
        train_dataset, _, test_dataset = prep_datasets(cfg, args)
        model = get_model('resnet18', num_classes=train_dataset.num_classes, pretrained=True)

        model.eval()
        model.cuda()
        print('Computing embedding for {}'.format(d))
        t2v_embedding = Task2Vec(model).embed(train_dataset)
        t2v_task_db[d] = {'hessian': torch.tensor(t2v_embedding.hessian),
                          'scale': torch.tensor(t2v_embedding.scale)}

        os.makedirs('expts/task_db', exist_ok=True)
        torch.save(t2v_task_db, args.output_path)

if __name__ == '__main__':
    parser = ArgumentParser('Extract task2vecs')
    parser.add_argument('--output_path', type=str, default='expts/task_db/resnet18_imagenet_20_tasks.pt')
    args = parser.parse_args()
    compute_task2vecs(args)


