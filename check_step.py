# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 16:16:15 2021

@author: DELL
"""

# check_step1.py (forward validation)
from reprod_log import ReprodDiffHelper

if __name__ == "__main__":
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("val/bp_param_torch.npy")
    paddle_info = diff_helper.load_info("val/bp_param_paddle.npy")
    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(path="val/bp_param_diff.log")