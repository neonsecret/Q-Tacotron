# -*- coding: utf-8 -*-
import os

import numpy as np
from tqdm import tqdm
from utils import load_spectrogram
import argparse

NUM_JOBS = 4


def f(fpath, orig_src):
    """
    Loads spectrograms from file paths and save as numpy

    :param fpath_orig_src: spectrogram file paths and texts
    """
    if fpath.split(".")[-1] in ["mp3", "wav", "opus"]:
        fname = orig_src + "_mels/" + fpath.split("/")[-1]
        if not os.path.isfile(fname) and not os.path.isfile(fname + ".npy"):
            mel, _ = load_spectrogram(fpath)
            np.save(fname, mel)
    return None


def preprocess(fpath):
    """
    Preprocess meta data and splits them for train and test set
    """
    print('Preprocessing meta')
    if fpath[-1] == "/":
        fpath = fpath[:-1]
    if not os.path.isdir(fpath + "_mels"):
        os.mkdir(fpath + "_mels")
    for fold1 in tqdm(os.listdir(fpath)):
        for fold2 in os.listdir(fpath + "/" + fold1):
            for audiofile in os.listdir(fpath + "/" + fold1 + "/" + fold2):
                f(fpath + "/" + fold1 + "/" + fold2 + "/" + audiofile, fpath)

    print('Complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mel loader.')
    parser.add_argument('fpath', type=str,
                        help='dataset filepath')

    args = parser.parse_args()
    preprocess(args.fpath)
