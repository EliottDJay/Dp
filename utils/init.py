import os
import sys
from datetime import datetime

import random
import numpy as np
import torch
from argparse import Namespace

from utils.logger import Logger as Log
from utils.basic_utils import mkdir

def init(args, is_inference = False):
    current_path = os.path.abspath(__file__)
    file_split = current_path.split('/')  # /./././.py
    path_new = os.path.join(* file_split[:-2])
    abspath = "/" + path_new

    if args.config:
        checkpoint_root = os.path.join(abspath, 'checkpoint')
        yaml_path = args.config
        path_split = yaml_path.split('/')  
        exp_dir = os.path.join(checkpoint_root, *path_split[-3:-1])  
        print(exp_dir)
        expiname = path_split[-3]+"_"+path_split[-2]
    else:
        NotImplementedError("No config file provided")

    model_path = os.path.join(exp_dir, "model")  
    summary_path = os.path.join(exp_dir, "summary")
    output_dir = os.path.join(exp_dir, "output") 
    data_analysis_dir = os.path.join(exp_dir, "data_analysis") 
    log_file = os.path.join(exp_dir, 'log') 
    # point and ckpt path
    point_path = os.path.join(model_path, "point_cloud")
    ckpt_path = os.path.join(model_path, "ckpt_point")

    mkdir([model_path, summary_path, output_dir, data_analysis_dir, log_file, point_path, ckpt_path])
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    if is_inference:
        path_dict = {
        "expiname": expiname,
        "exp_path": exp_dir,
        "model_path": model_path,
        "summary_path": summary_path,
        "output_dir": output_dir,
        "data_analysis_dir": data_analysis_dir,
        "log_file": os.path.join(log_file, expiname + current_time + "inference_render.log"),
    }
    else:
        path_dict = {
        "expiname": expiname,
        "exp_path": exp_dir,
        "model_path": model_path,
        "summary_path": summary_path,
        "output_dir": output_dir,
        "data_analysis_dir": data_analysis_dir,
        "log_file": os.path.join(log_file, expiname + current_time + ".log"),
        }
    setattr(args, "path_dict", path_dict)

    # log inital
    Log.init(logfile_level=args.logfile_level, stdout_level=args.stdout_level, log_file=path_dict["log_file"],
             log_format=args.log_format, rewrite=args.rewrite)

    Log.info("Checkpoint root folder: {} ; expi name: {}".format(exp_dir, expiname))

    if is_inference:
        with open(os.path.join(exp_dir, "cfg_args"), 'w') as cfg_log_f:
            cfg_log_f.write(str(Namespace(**vars(args))))
        with open(os.path.join(exp_dir, 'command_line.txt'), 'w') as file:
            file.write(' '.join(sys.argv))

    seed = args.seed
    init_seeds(seed, cuda_deterministic=args.deterministic)

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cuda_deterministic: 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False  
        torch.backends.cudnn.benchmark = False  
    else:  
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

