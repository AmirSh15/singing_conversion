import logging
import os
import re
import sys
import urllib.request

from pydub import AudioSegment
from pytube import YouTube
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_search_result(keyword: str) -> list:
    """_summary_
        A function to get a search keyword as input and request the search result from youtube

    Args:
        keyword (string): the keyword to search

    Returns:
        list: a list of video ids
    """

    # get the search result from youtube
    html = urllib.request.urlopen(
        "https://www.youtube.com/results?search_query=" + keyword
    )
    video_ids = re.findall(r"watch\?v=(\S{11})", html.read().decode())
    return video_ids


def get_search_result_advanced(keyword: str, num_pages: int = 1) -> list:
    """_summary_
        A function to get a search keyword as input and request the search result from youtube
        using youtube-search-python package

    Args:
        keyword (string): the keyword to search
        num_pages (int, optional): the number of pages to search. Defaults to 1.

    Returns:
        list: a list of video ids
    """

    # import the youtube-search-python package
    from youtubesearchpython import VideosSearch

    # initialize the video ids list
    video_ids = []
    # get the search result from youtube
    videosSearch = VideosSearch(keyword, limit=20)
    for i in range(num_pages):
        video_ids.extend([video["id"] for video in videosSearch.result()["result"]])

        # get the next page
        if i < num_pages - 1:
            videosSearch.next()

    return video_ids


def download_audio(video_id: str, output_path: str, max_size: int = 10) -> None:
    """_summary_
        A function to get the youtube video url and download its audio with pytube

    Args:
        video_id (string): the video id of the youtube video
        output_path (string): the path to save the audio file
        max_size (int, optional): the maximum size of the audio file in MB. Defaults to 100.
    """

    # check if the output path exists
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # get the youtube video url
    url = "https://www.youtube.com/watch?v=" + video_id

    # download the audio
    yt = YouTube(url)
    audio = yt.streams.filter(only_audio=True).asc().first()
    audio_ext = audio.mime_type.split("/")[-1]
    audio_name = audio.default_filename.split(".")[0]

    # read file size
    audio_size = audio.filesize
    # convert to MB
    audio_size = audio_size / (1024 * 1024)

    # check if the audio file size is less than max_size
    if audio_size < max_size:
        # check if the audio file already exists
        if os.path.exists(os.path.join(output_path, video_id + ".wav")):
            # logger.info("The audio file already exists")
            return

        # save the audio file
        audio.download(output_path=output_path)

        # if the audio extension is not wav, convert it to wav
        if audio_ext != "wav":
            # read the audio file
            sound = AudioSegment.from_file(
                os.path.join(output_path, audio.default_filename), format=audio_ext
            )

            # export the audio file as wav
            # sound.export(os.path.join(output_path, audio_name + ".wav"), format="wav")
            sound.export(os.path.join(output_path, video_id + ".wav"), format="wav")

            # remove the original audio file
            os.remove(os.path.join(output_path, audio.default_filename))


def download_audio_results(keyword: str, output_path_root: str, num_pages=1) -> None:
    """_summary_
        A function to get a keyword as input and download the audio of the first search result

    Args:
        keyword (string): the keyword to search
        output_path_root (string): the path to save the audio file
        num_pages (int, optional): the number of pages to search. Defaults to 1.
    """

    # get the search result
    logger.info("Getting the search result for keyword: {}".format(keyword))
    video_ids = get_search_result_advanced(keyword, num_pages=num_pages)

    # a text file to store the failed video ids
    failed_video_ids = []
    # download the audio files
    for video_id in tqdm(
        video_ids, desc="Downloading audio files", total=len(video_ids)
    ):
        try:
            output_path = os.path.join(output_path_root, keyword)
            download_audio(video_id, output_path)
        except Exception as e:
            failed_video_ids.append(video_id)
            continue

    # save the failed video ids
    with open(os.path.join(output_path_root, "failed_video_ids.txt"), "w") as f:
        f.write("\n".join(failed_video_ids))
