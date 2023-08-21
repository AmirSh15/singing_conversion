import argparse
import glob
import os

from pydub import AudioSegment
from tqdm import tqdm

AUDIO_EXTENSIONS = ["mp3", "mp4"]


def convert_to_wav(input_path: str, output_path: str) -> None:
    """_summary_

    Args:
        input_path (string): the path of the directory that contains the audio files
        output_path (string): the path of the directory to save the converted audio files
    """

    # get the audio files in mp3, mp4 in the input path
    audio_files = []
    for ext in AUDIO_EXTENSIONS:
        audio_files += glob.glob(os.path.join(input_path, "*." + ext))

    # convert the audio files to wav format
    for audio_file in tqdm(
        audio_files, desc="Converting audio files to wav format", total=len(audio_files)
    ):
        # read the audio file
        sound = AudioSegment.from_file(audio_file, format="mp3")

        # export the audio file as wav
        sound.export(
            os.path.join(
                output_path, os.path.basename(audio_file).split(".")[0] + ".wav"
            ),
            format="wav",
        )

        # remove the original audio file
        os.remove(audio_file)


if __name__ == "__main__":
    # get the arguments from the command line
    parser = argparse.ArgumentParser(
        description="A program to convert text to singing voice"
    )
    parser.add_argument(
        "--path",
        type=str,
        default="./so-vits-svc/dataset_raw",
        help="the path of the directory that contains the audio files",
    )
    args = parser.parse_args()

    # convert the audio files to wav format
    convert_to_wav(args.path, args.path)
