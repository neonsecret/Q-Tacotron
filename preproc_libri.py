# -*- coding: utf-8 -*-
import os
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm
from utils import load_spectrogram
import argparse

NUM_JOBS = 4


def f(hyperarg):
    """
    Loads spectrograms from file paths and save as numpy

    :param hyperarg: spectrogram file paths and root folder path
    """
    fpath, orig_src = hyperarg
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

    arglist = []

    for fold1 in os.listdir(fpath):
        for fold2 in os.listdir(fpath + "/" + fold1):
            for audiofile in os.listdir(fpath + "/" + fold1 + "/" + fold2):
                arglist.append((fpath + "/" + fold1 + "/" + fold2 + "/" + audiofile, fpath))

    with Pool(NUM_JOBS) as p:
        with tqdm(total=len(arglist)) as pbar:
            for _ in p.imap_unordered(f, arglist):
                pbar.update()

    print('Complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mel loader.')
    parser.add_argument('fpath', type=str,
                        help='dataset filepath')

    args = parser.parse_args()
    preprocess(args.fpath)
