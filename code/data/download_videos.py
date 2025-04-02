import requests
import os
import pandas as pd
import cv2

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
        
        print(f"\tVideo successfully downloaded to {output_path}")
    
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

def download_videos_from_metadata(metadata: pd.DataFrame, output_path: str, verify_ssl_settings: dict = None, verbose: bool = True) -> None:
    """
    Download all videos from the metadata DataFrame.
    Args:
        metadata (pd.DataFrame): DataFrame containing metadata for videos with columns: label, data_source, video_url
        verify_ssl_settings (dict): Dictionary mapping data source codes to SSL verification settings.
                                  e.g., {'ne': True, 'vl': False, 'sb': False, 'uf': True}
        verbose (bool): If True, print download status messages.
    """
    if metadata.empty:
        print("No data found in metadata")
        return
    
    # Default SSL verification settings if none provided
    if verify_ssl_settings is None:
        verify_ssl_settings = {code: True for code in data_source_codes.keys()}
    
    # Loop through each row in the metadata
    for label, label_metadata in metadata.groupby('label'):
        for index, row in label_metadata.reset_index().iterrows():
            video_url = row['video_url']
            video_name = make_video_filename(row, index + 1)
            
            video_path = os.path.join(output_path, video_name)
        
        if verbose:
            print(f"Downloading video {index + 1} from {video_url}")
        
        # Get SSL verification setting for this data source
        verify_ssl = verify_ssl_settings.get(row['data_source'], True)
        download_video_from_link(video_url, video_path, verify_ssl=verify_ssl)

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
        "duration_sec": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
    }
    
    cap.release()
    return metadata

def collect_metadata_from_directory(directory):
    video_files = sorted([f for f in os.listdir(directory) if f.endswith(".mp4")])
    all_metadata = [get_video_metadata(os.path.join(directory, f)) for f in video_files]

    return all_metadata