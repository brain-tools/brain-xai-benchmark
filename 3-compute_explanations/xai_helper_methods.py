import numpy as np
import json
import pickle
import os
import json    

def load_masker(masker_path):
    with open(masker_path, 'rb') as file:
        nifti_masker = pickle.load(file)
    return nifti_masker

def create_result_dir(res_path):
    # create result directory
    isExist = os.path.exists(res_path)
    if not isExist:
        os.makedirs(res_path)

def init_xai_dict(split_path, xai_methods, mask_size):
    xai_dict = {}
    with open(split_path, 'r') as file:
        split = json.load(file)

    n_test = len(split["test"])

    for xai_m in xai_methods:
        mean_expl = np.zeros((182, 218, 182))
        # for aseg:
        masked_expl = np.zeros((n_test, mask_size), dtype="float16")
        eids = np.zeros((n_test))

        m_dict = {"mean_expl" : mean_expl, "cont_dice" : [], "masked_expl":masked_expl, "eids":eids}
        xai_dict[xai_m] = m_dict

    return xai_dict, n_test