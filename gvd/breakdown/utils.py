import os
import scanpy as sc
from gears.utils import zip_data_download_wrapper

def data_download(data_name,data_dir,extract_dir):
    if data_name == 'norman':
        url = 'https://dataverse.harvard.edu/api/access/datafile/6154020'
    elif data_name == 'adamson':
        url = 'https://dataverse.harvard.edu/api/access/datafile/6154417'
    elif data_name == 'dixit':
        url = 'https://dataverse.harvard.edu/api/access/datafile/6154416'
    elif data_name == 'replogle_k562_essential':
        ## Note: This is not the complete dataset and has been filtered
        url = 'https://dataverse.harvard.edu/api/access/datafile/7458695'
    elif data_name == 'replogle_rpe1_essential':
        ## Note: This is not the complete dataset and has been filtered
        url = 'https://dataverse.harvard.edu/api/access/datafile/7458694'
    else:
        print("None of these datasets exist")
        return
    zip_data_download_wrapper(url, data_dir, extract_dir)
    

def lfc_weighted_