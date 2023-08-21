import argparse
import glob
import os
import librosa  # Optional. Use any library you like to read audio files.
import soundfile
from tqdm import tqdm  # Optional. Use any library you like to write audio files.

from slicer2 import Slicer
    
def main():
    
    # get the arguments from the command line
    parser = argparse.ArgumentParser(description="A program to slice audio files")
    parser.add_argument(
        "--raw_audio_path",
        type=str,
        default="./so-vits-svc/raw",
        help="the path of the directory that contains the audio files",
    )
    parser.add_argument(
        "--raw_vocal_audio_path",
        type=str,
        default="./so-vits-svc/raw_vocal",
        help="the path of the directory that contains the vocal audio files",
    )
    parser.add_argument(
        "--processed_audio_path",
        type=str,
        default="./so-vits-svc/dataset_raw",
        help="the path of the directory to save the processed audio files",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=20,
        help="the maximum length of the sliced audio files in seconds",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=2,
        help="the minimum length of the sliced audio files in seconds",
    )
    args = parser.parse_args()
    
    # loop over directories in the raw audio path
    for dir in os.listdir(args.raw_audio_path):
        # check dir is actually a directory
        if not os.path.isfile(os.path.join(args.raw_audio_path, dir)):
        
            # find the audio files
            audio_files = glob.glob(os.path.join(args.raw_audio_path, dir, "*.wav"))
            
            # create the directory to save the sliced audio files
            if not os.path.exists(os.path.join(args.processed_audio_path, dir)):
                os.makedirs(os.path.join(args.processed_audio_path, dir))
            
            # read and slice the audio files
            for audio_file in tqdm(
                audio_files, desc=f"Slicing audio files in {os.path.join(args.raw_audio_path, dir)}", total=len(audio_files)
            ):
                audio_file_name = os.path.basename(audio_file).split(".")[0]
                
                # find the corresponding vocal audio file
                vocal_audio_file = glob.glob(os.path.join(args.raw_vocal_audio_path, dir, f"*{audio_file_name}*"))[0]
                
                # read the audio files
                audio, sr = librosa.load(audio_file, sr=None, mono=False)
                vocal_audio, vocal_sr = librosa.load(vocal_audio_file, sr=None, mono=False)
                
                # slice the audio files
                slicer = Slicer(
                    sr=sr,
                    threshold=-30,
                    min_length=5000,
                    min_interval=300,
                    hop_size=10,
                    max_sil_kept=500
                )
                
                # slice the audio file
                chunks = slicer.slice(audio, vocal_audio)
                
                # save the sliced audio files
                split = 0
                for i, chunk in enumerate(chunks):
                    if len(chunk.shape) > 1:
                        chunk = chunk.T
                    # break down the audio file if it is longer than the maximum length
                    if len(chunk) > args.max_length * sr:
                        for j in range(0, len(chunk), args.max_length * sr):
                            # skip segments with less than min_length seconds
                            if len(chunk[j:j + args.max_length * sr]) < args.min_length * sr:
                                continue
                            soundfile.write(os.path.join(args.processed_audio_path, dir, f'{audio_file_name}_{split}.wav'), chunk[j:j + args.max_length * sr], sr)
                            split += 1
                    else:
                        # skip segments with less than min_length seconds
                        if len(chunk) < args.min_length * sr:
                            continue
                        soundfile.write(os.path.join(args.processed_audio_path, dir, f'{audio_file_name}_{split}.wav'), chunk, sr)  # Save sliced audio files with soundfile.
                        split += 1
                
                
if __name__ == '__main__':
    main()