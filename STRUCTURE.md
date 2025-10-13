# Project Structure

```
SaoPauloBrazilChapter_BrazilianSignLanguage/
├── data/                      # Data files
│   ├── raw/                  # Original data
│   │   ├── INES/           # INES dataset
│   │   │   └── videos/    # Video files (stored on Google Drive)
│   │   ├── SignBank/      # SignBank dataset
│   │   │   └── videos/   # Video files (stored on Google Drive)
│   │   ├── UFV/          # UFV dataset
│   │   │   └── videos/  # Video files (stored on Google Drive)
│   │   └── V-Librasil/  # V-Librasil dataset
│   │       └── videos/ # Video files (stored on Google Drive)
│   ├── interim/             # Intermediate data
│   │   ├── Analysis/        # Analysis info of the data
│   │   │   └── [analysis_info.json files]
│   │   ├── Debug/         # Debug files for inspecting preprocessing steps
│   │   │   ├── videos/   # Intermediate video files
│   │   │   └── landmarks/ # Intermediate landmark files
│   │   ├── RawCleanVideos/ # The raw video files after some initial cleaning before the preprocessing pipeline
│   │   ├── RawMotionMeasurements/ # Raw motion measurements
│   │   └── RawPoseLandmarks/      # Raw pose landmarks
│   ├── preprocessed/           # Final datasets
│   │   ├── metadata_v*.csv # Metadata for each preprocessing version
│   │   ├── videos/      # Preprocessed videos
│   │   │   └── v*/      # Version-specific processed videos (e.g., v1, v2, v3...)
│   │   └── landmarks/   # Processed landmark data
│   │       └── v*/      # Version-specific processed landmarks (e.g., v1, v2, v3...)
│   ├── external/            # Third party sources
│   ├── papers/             # Related research papers
│   └── README.md           # Data documentation
│
├── docs/                   # Files for the github pages website
│   └── assets/             # Images, gifs, etc.
│
├── code/                  # Source code
│   ├── data/              # Data scraping & cleaning
│   ├── eda/               # Exploratory data analysis
│   ├── model/             # Model implementation & training
│   └── preprocess/        # Preprocessing pipeline
│
├── modelling/          # Model training and evaluation
│   ├── logs/
│   ├── metadata/
│   └── runs
│
├── webapp/          # Webapp
│   ├── backend/
│   └── frontend/
│
├── pyproject.toml          # Project metadata and dependencies
├── uv.lock                # Locked dependencies
├── README.md              # Project documentation
└── STRUCTURE.md           # This file
```

## Data Management

### Video Storage
- Large video files are stored on Google Drive, not tracked in Git
- Video directories in the repository are placeholders
- Download videos to your local `videos/` directories as needed
- Preprocessing pipeline creates versioned outputs (v1, v2, v3...) of processed videos and landmarks
- Each version has its own metadata and individual file metadata

### Data Files
- Small files like CSV files, labels, and metadata are tracked in Git
- Processed data (features, embeddings) stored in `preprocessed/`
- Document data formats in respective directories
- Metadata files track preprocessing steps and configurations

### Python Environment
- Managed by `uv` package manager
- Dependencies specified in `pyproject.toml`
- Versions locked in `uv.lock`

## Directory Details

### `data/`
- `raw/`: Original, immutable data
  - Dataset directories (INES, SignBank, UFV, V-Librasil)
  - Each dataset has a `videos/` subdirectory (videos on Google Drive)
  - CSV files and labels tracked in Git
- `interim/`: Intermediate processed data
  - `Debug/`: Files for inspecting preprocessing steps
  - `RawMotionMeasurements/`: Raw motion detection results
  - `RawPoseLandmarks/`: Raw pose detection results
  - `Videos/`: Preprocessed video files
- `preprocessed/`: Final, model input datasets
  - Version-specific directories (v1, v2, v3...)
  - Separate directories for videos and landmarks
  - Metadata files for each version
  - Individual metadata files for each processed file

### `code/`
- `data/`: Data scraping & cleaning
- `eda/`: Exploratory data analysis
- `model/`: Model implementation & training
- `preprocess/`: Preprocessing pipeline

### `modelling/`
- `logs/`: Logs
- `metadata/`: Metadata
- `runs/`: Runs