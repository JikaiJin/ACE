import sys
import os
import argparse
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(args):
    os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
    parser = argparse.ArgumentParser(
        description="Second order orthogonal ML!")
    parser.add_argument("--input_dir",  dest="input_dir", type=str, default=".")
    parser.add_argument("--output_dir", dest="output_dir", type=str, default=".")
    opts = parser.parse_args(args)

    if not os.path.exists(opts.output_dir):
        os.makedirs(opts.output_dir)

    
if __name__=="__main__":
    main(sys.argv[1:])
