
import requests
import os
import pandas as pd
import cv2

data_source_codes = {
    'ne': 'INES',
    'vl': 'V-Librasil',
    'sb': 'SignBank',
    'uf': 'UFV'
}

def download_video_from_link(link: str, output_path: str, verify_ssl: bool = True) -> None:
    """
    Download a video from a URL and save it to a file.
    Args:
        link (str): The URL of the video to download.
        output_path (str): The location where the video will be saved.
        verify_ssl (bool): Whether to verify SSL certificates. Set to False for self-signed/invalid certs.
    """
    try:
        # Make a GET request to fetch the video content
        response = requests.get(link, stream=True, verify=verify_ssl)
        response.raise_for_status()  # Raise an error for bad responses

        # Open the output file in write-binary mode
        with open(output_path, 'wb') as video_file:
            for chunk in response.iter_content(chunk_size=8192):
                video_file.write(chunk)
        
        print(f"Video successfully downloaded to {output_path}")
    
    except requests.exceptions.RequestException as e:
        print(f"Error downloading video from {link}: {e}")

def make_video_filename(row: pd.Series, index: int) -> str:
    """
    Make a video filename from the row and index.
    Args:
        row (pd.Series): The row of the metadata.
        index (int): The index of the video.
    Returns:
        str: The filename of the video.
    """
    return f"{row['label']}_{row['data_source']}_{index}.mp4"

def download_videos_from_metadata(label: str, metadata: pd.DataFrame, data_source_key: str = None, combined: bool = False, verbose: bool = True, verify_ssl: bool = True) -> None:
    """
    Download all videos for one word from one data source.
    Args:
        label (str): The label of the word/video to download.
        metadata (pd.DataFrame): DataFrame containing metadata for videos (e.g., URLs).
        data_source_key (str): The key of the data source to download from. (e.g. 'ne', 'vl', 'sb', 'uf') If None, all data sources will be downloaded from.
        combined (bool): If True, the videos will be downloaded into the /raw/combine/videos folder.
        verbose (bool): If True, print download status messages.
        verify_ssl (bool): Whether to verify SSL certificates. Set to False for self-signed/invalid certs.
    """
    # Filter metadata for the given data source
    if data_source_key is not None:
        filtered_metadata = metadata[metadata['data_source'] == data_source_key]
    else:
        filtered_metadata = metadata
    
    if filtered_metadata.empty:
        print(f"No data found for source: {data_source_codes[data_source_key]}")
        return
    
    # Filter for the given label
    filtered_metadata = filtered_metadata[filtered_metadata['label'] == label]
    if filtered_metadata.empty:
        print(f"No data found for label: {label} in {data_source_codes[data_source_key]}")
        return
    
    # Loop through each row in the filtered metadata
    i = 1
    for df_index, row in filtered_metadata.iterrows():
        video_url = row['video_url']
        video_name = make_video_filename(row, i)

        if combined:
            output_path = os.path.join('data', 'raw', 'combined', 'videos')
        else:
            if data_source_key is None:
                data_source = data_source_codes[row['data_source']]
            else:
                data_source = data_source_codes[data_source_key]
            output_path = os.path.join('data', 'raw', data_source, 'videos')
        video_path = os.path.join(output_path, video_name)
        
        if verbose:
            print(f"Downloading video {i} from {video_url}")
        # download_video_from_link(video_url, video_path, verify_ssl=verify_ssl)
        if data_source_key == 'sb':
            download_video_from_link(video_url, video_path, verify_ssl=False)
        else:
            download_video_from_link(video_url, video_path, verify_ssl=verify_ssl)
        i += 1


def get_video_metadata(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    metadata = {
        "filename": os.path.basename(video_path),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration_sec": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
    }
    
    cap.release()
    return metadata

def collect_metadata_from_directory(directory):
    video_files = sorted([f for f in os.listdir(directory) if f.endswith(".mp4")])
    all_metadata = [get_video_metadata(os.path.join(directory, f)) for f in video_files]

    return all_metadata