import argparse
import glob
import logging
import os

from tqdm import tqdm

from vocal_remover_utils import SeperateVR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# remove all the warnings
import warnings
warnings.filterwarnings("ignore")


def get_args():
    """_summary_
        A function to get the arguments from the command line

    Returns:
        argparse.Namespace: a namespace of the arguments
    """

    parser = argparse.ArgumentParser(description='A program to remove vocal from audio files')
    parser.add_argument('--input_path', type=str, default='/home/so-vits-svc/raw', help='the path to the audio files')
    parser.add_argument('--output_path', type=str, default='/home/so-vits-svc/raw_vocal', help='the path to save the audio files')
    parser.add_argument('--model', type=str, default='VR_Arch', help='the current option is VR_Arch')
    args = parser.parse_args()

    return args
            
def vocal_remover():
    # get the arguments from the command line
    args = get_args()
    
    seperator = SeperateVR()
    
    # read the audio files
    for singer in os.listdir(args.input_path):
        # check if the singer is a directory
        if  os.path.isdir(os.path.join(args.input_path, singer)):
            songs = os.listdir(os.path.join(args.input_path, singer))
            for song in tqdm(songs, total=len(songs), desc=f'Seperating the audio files of {singer}'):
                # check if the song is already seperated
                song_name = song.split('.')[0]
                vocal_audio_file = glob.glob(os.path.join(args.output_path, singer, f"{song_name}*"))
                if len(vocal_audio_file) == 0:
                    seperator.seperate(
                        os.path.join(args.input_path, singer, song),
                        os.path.join(args.output_path, singer),
                        vocal_or_instrument='vocal',
                    )
                                    
                                    
if __name__ == '__main__':
    vocal_remover()