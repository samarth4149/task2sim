from typing import List

import torch

from generator.generator_params import GenParams


def add_categorical_noise(a: torch.Tensor, eps: float):
    # Entries of a should sum to 1
    eps = min(eps, a.max())
    a[a.argmax()] -= eps + eps/(len(a) - 1)
    a += eps/(len(a) - 1)
    return a

def sample_w_noise_gen_params_from_outputs(batch_size, cont_outputs, eps=0.1) -> List[GenParams]:
    gps = []
    for i in range(batch_size):
        gps.append(
            GenParams(
                {k:torch.multinomial(
                    add_categorical_noise(torch.softmax(v[i], 0), eps), 1).item()
                 for k,v in cont_outputs.items()}))
    return gps

def post_process_for_mat(gps:List[GenParams]):
    for i in range(len(gps)):
        gps[i]['materials'] = 2*gps[i]['materials'] + 1
    return gps

def sample_gen_params_from_outputs(batch_size, cont_outputs) -> List[GenParams]:
    gps = []
    for i in range(batch_size):
        gps.append(GenParams({k:torch.multinomial(torch.softmax(v[i], 0), 1).item() for k,v in cont_outputs.items()}))
    if 'materials' in cont_outputs:
        post_process_for_mat(gps)
    return gps

def sample_gen_params_from_outputs_softmax(batch_size, cont_outputs) -> List[GenParams]:
    gps = []
    for i in range(batch_size):
        gps.append(GenParams({k:torch.multinomial(v[i], 1).item() for k,v in cont_outputs.items()}))
    if 'materials' in cont_outputs:
        post_process_for_mat(gps)
    return gps


def get_argmax_gen_params_from_outputs(batch_size, cont_outputs)  -> List[GenParams]:
    gps = []
    for i in range(batch_size):
        gps.append(GenParams({k:v[i].argmax().item() for k,v in cont_outputs.items()}))
    if 'materials' in cont_outputs:
        post_process_for_mat(gps)
    return gps