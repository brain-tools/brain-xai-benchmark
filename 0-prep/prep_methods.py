import pandas as pd
import pickle
from pathlib import Path


def creat_idp_df_pkl(idp_identifier_path='/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/setup_files/idp_fields.pkl',
                     idp_path='/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/setup_files/t1-idps.tsv',
                     ukb_table_path='/sc-resources/ukb/data/projects/33073/ukb_data/table/ukb_imaging_filtered.txt', 
                     save_path = "/sc-projects/sc-proj-cc15-cn-ukbiobank/analyses/Explanation-benchmark-paper/files/setup_files/idp_df.pkl",
                     subset_of_idp_identifier_path=None):
    
    # load idp identifier
    with open(idp_identifier_path, 'rb') as handle:
        idp_fields = pickle.load(handle)

    # load idp data
    idp_df = pd.read_csv(idp_path, sep='\t', names=idp_fields)

    if subset_of_idp_identifier_path is not None:
        # load subset of identifier
        with open(subset_of_idp_identifier_path, 'rb') as handle:
            idp_fields_subset = pickle.load(handle)
            

        idp_df = idp_df[idp_fields_subset]

    # load respective eids
    columns = ["f.eid"]

    eids = pd.read_csv(ukb_table_path, on_bad_lines='skip', sep="\t",                                                                        
                        usecols=columns)

    idp_df["eid"] = eids
    idp_df.to_pickle(save_path)

    return idp_df

def img_paths_to_idp_df_eid_in_fname(path_to_images, idp_df, n_imgs):
    # First, read filenames of lin. registered images (see save_linearly_registered_images.py)
    img_paths = []

    p = Path(path_to_images)
    for sub_folder in p.iterdir():
        if "eid" not in str(sub_folder):
            img_paths.append(path_to_images + "/" + sub_folder.name)

    reg_img_df = pd.DataFrame(columns = ["eid", "reg_img_path"])

    for i, img_path in enumerate(img_paths):
        if i < n_imgs:
            eid = int(img_path.split("/")[-1].split(".")[0])
            reg_img_df.loc[len(reg_img_df)] = {"eid":eid, "reg_img_path":img_path}

    # keep non image idps:
    idp_df = pd.merge(idp_df, reg_img_df, on='eid', how="left")

    return idp_df