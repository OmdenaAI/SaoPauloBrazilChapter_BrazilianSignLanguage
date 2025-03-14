# Data Directory

This directory contains data files for the Brazilian Sign Language Recognition project. Small files like .csv's with metadata can be tracked by git. Large files like videos should be stored on Google Drive.

## Directory Structure

- `raw/` - Original, immutable data
  - Video recordings of sign language
  - Any external datasets
  
- `interim/` - Intermediate data that has been transformed
  - Extracted frames
  - Preprocessed videos
  
- `processed/` - Final, model input datasets
  - Feature vectors
  - Processed landmarks
  - Training/validation/test splits

## Information about `raw/`

### Data Sources

- There are 4 sources we are getting data from:
  - INES
  - SignBank
  - UFV
  - V-Librasil
- In `raw/`, data for each will be stored in a subfolder `raw/{data-source/}`.
- For short, we will sometimes use 2 character codes for each data source:
  - `ne`
  - `sb`
  - `uf`
  - `vl`


### Metadata

- Apart from the actual video files, `metadata.csv` files containing all info about each dataset will created
- Each data source will share some common columns with key info that will be needed later
- Each data source will also have some unique columns, containing info specific to that dataset
- If adding new columns /info to a `metadata.csv`:
  - Ideally save it as an independent file first, e.g. `extra_info.csv`
  - Then write code in `raw_metadata.ipynb` to add the information into `metadata.csv`
  - This way:
     1. We won't lose information by directly edition `metadata.csv`
     2. We can reproduce `metadata.csv` from the source information

### Common Columns
The following columns should be present in all metadata.csv files regardless of data source:

- `label` - The label associated with the video. Could be a letter, word, or phrase depending on the source.
- `video_url` - URL to the downloadable video file
- `signer_number` - Identifier for the person performing the sign in the video.
  - Sometimes taken directly from the source e.g. V-Librasil
  - Sometimes assigned by us e.g. SignBank

### INES Dataset
Columns in INES metadata:
- `scraped_label` - The originally scraped label. 
  - Some labels were unified e.g. 'MAIS1' & 'MAIS2' 
- `file_exists` - True/False to indicate if the video is accessible at `video_url`
  - Some INES videos are inaccessible due to INES's own issues.
  - Each `video_url` was checked in `raw_metadata.ipynb` and this column was made
- `letter` - Starting letter of the word
- `assuntos` - Subject/topic categories
- `acepção` - Definition/meaning of the word
- `exemplo` - Example sentence in Portuguese
- `exemplo libras` - Example sentence (in Libras notation?)
- `classe gramatical` - Grammatical class
- `origem` - Origin of the sign
- `number_in_label` - True/False column indicating if the `scraped_label` has a number in it.
  - used to process `scraped_label`s into `label`s

### V-Librasil Dataset
Columns in V-Librasil metadata:
- `sign_url` - URL to the sign's webpage which contains all videos
- `signer_order` - 3-digit number indicating the order of signers on the sign's webpage

### SignBank Dataset
Columns in V-Librasil metadata:
- `scraped_label` - The originally scraped label. 
  - Some labels were unified e.g. 'ABORTAR' & 'ABORTAR-2' 
- `scraped_video_url` - The originally scraped video url. Some invalid url's were fixed to make `video_url`
- `sign_variant` - Integer indicating which sign variant a video is, starting from `1`.
  - Some labels have multiple videos (from the same signer).
  - e.g. 'ABORTAR' -> `1` & 'ABORTAR-2' -> `2`
  - They might be different sign variants for the same word
  - They might be homographs- different words that are spelled the same
- `signer_number` - Integer indicating which signer is performing in the video.
  - The large majority are from signer `1`, who has facial hair, an earing, and a black button-up shirt
  - A minority are from signer `2`, who has no facial hair, and a black t-shirt
  - (as of 12th March, it is based on some assumptions, and not verified for all videos - Ben)
  - (my assumption is that all videos are signer `1`, except those with `video_url_root` = `levantelab`)
  - (I also saw a third signer, but I think the videos are very rare, and i'm not sure the pattern)
  - `video_url_root` - string from the first part of the `scraped_video_url`
    - 2992 'videos', 63 'objectos', 28 'levantelab'
  - `video_url_ext` - string with the file extension in `video_url`
    - 3063 'mp4', 21 'mov'
  - `number_in_label` - True/False column indicating if the `scraped_label` has a number in it.
    - used to process `scraped_label`s into `label`s