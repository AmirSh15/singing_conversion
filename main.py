import argparse
import os

from utils.utils import download_audio_results
from utils.vocal_remover import vocal_remover
from utils.audio_slicer import audio_slice

# include "so-vits-svc" as python path
import sys

from tqdm import tqdm
sys.path.append('/home/so-vits-svc/')
from resample import resample
from preprocess_first_config import make_config
from preprocess_hubert_f0 import preprocess

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_args():
    """_summary_
        A function to get the arguments from the command line

    Returns:
        argparse.Namespace: a namespace of the arguments
    """

    parser = argparse.ArgumentParser(description='A program to convert text to singing voice')
    parser.add_argument('--keywords', type=str, nargs='+', default=['adele', 'micheal jakson'], help='the name of signers to search')
    parser.add_argument('--num_pages', type=int, default=5, help='the number of pages to search')
    parser.add_argument('--output_path', type=str, default='./so-vits-svc/raw', help='the path to save the audio files')
    parser.add_argument('--advanced', type=bool, default=False, help='whether to use the advanced slicing method')
    args = parser.parse_args()

    return args

def main():
    
    # get the arguments from the command line
    args = get_args()
    
    # check if the output path exists
    if not os.path.exists(args.output_path):
        raise Exception("The output path does not exist")
    
    # download the audio files for singers in the keywords list
    for keyword in args.keywords:
        download_audio_results(keyword, args.output_path, args.num_pages)
        
    # remove the vocal from the audio files
    if args.advanced:
        vocal_remover()
    
    # slice the audio files
    audio_slice(advanced=args.advanced)
    
    # resample the audio files to 44.1 kHz
    resample()
    
    # make the config files (train and val split)
    make_config()
    
    # Generate hubert and F0 features
    preprocess()
        
if __name__ == '__main__':
    main()