<mark>This page is under construction</mark>

# Healthcare Sign Language Brazil


## Summary
*This should be somewhat similar to what is described in the [project's main page](https://www.omdena.com/chapter-challenges/ai-assisted-sign-language-translation-for-brazilian-healthcare-settings)*

### Goal

### Contributors
Here's a list of people that contributed to the project. Feel free to reach out to them if needed:

[Ben Thompson](https://www.linkedin.com/in/ben-d-thompson/) (Project Manager)

[Ayushya Pare](https://www.linkedin.com) (Data Scraping)

[Gustavo Santos](https://www.linkedin.com/in/gustavopsantos) (Data cleaning)

[Anastasiia Derzhanskaia](https://www.linkedin.com) (Model development)

[Kunal Sood](https://www.linkedin.com) (Model development)

[Patrick Fitz](https://www.linkedin.com) (App development)

[Damla Helvaci](https://www.linkedin.com)

[Ethel Phiri](https://www.linkedin.com/in/ethel-phiri261312/)

[Gulzar](https://www.linkedin.com)

[Michael Spurgeon Jr](https://www.linkedin.com)

[Pooja Prasad](https://www.linkedin.com)

[Lorem Ipsum](https://www.linkedin.com) (Main role)


### Results
*Briefly describe main results and use a [link] (#results-1) reference somewhere in your text to send the reader to a complete version of the results* 

### Demo app
*Briefly describe the app and use a [link] (#demo-app-1) reference somewhere in your text to send the reader to a more complete description of the app* 

## Introduction
### Problem statement
*Similar to what we could find in the Omdenas website*

### Sign language processing
*We could do a brief description of the different tasks that are possible with SLP*
*Also, reference the site we used during research phase*

### Research steps
*Cite the paper we used as reference*

## Data

### Scraping

To build a robust dataset for Brazilian Sign Language (Libras), we identified four sources containing hundreds of videos representing various signs. Due to the scale and structure of these sources, we implemented web automation tools—primarily **Selenium**—to efficiently extract video URLs and relevant metadata.

The scraped data was compiled into a CSV file, which served as a foundational resource for later stages of data cleaning, preprocessing, and modeling.

**Data Sources:**

* [INES](https://www.ines.gov.br/dicionario-de-libras/)
* [Signbank (UFSC)](https://signbank.libras.ufsc.br/pt)
* [UFV Dictionary](https://sistemas.cead.ufv.br/capes/dicionario/)
* [V-Librasil Dataset](https://ieee-dataport.org/documents/v-librasil-new-dataset-signs-brazilian-sign-language-libras#files)


We scraped data from three Brazilian Sign Language (Libras) resources: INES, Signbank (UFSC), and VLibras. We used Python with Selenium to automate browsing and extract videos, words, and metadata. Each website had a different structure, so we customized the code for each. On INES, videos were downloaded directly. UFSC’s Signbank had scattered links, which we cleaned using pandas. VLibras was slower and more complex, so we scraped it manually with help from browser tools. We saved all data in organized CSV files.

**Web Automation Approach:**
We used Selenium to automate the collection process, enabling consistent extraction of video files and associated metadata across varying site structures.

**Extracted Metadata Fields:**
The specific metadata varied slightly by source, but key fields included:

* `alphabet_label`: the starting letter of the signed word
* `label`: the word being signed in the video
* `video_url`: direct link to the video file

This standardized structure allowed us to create a unified dataset for downstream processing.

### Cleaning
### Review steps
### Final dataset

## Preprocessing
### EDA
### Pose estimation with [MediaPipe Holistic](https://ai.google.dev/edge/mediapipe/solutions/guide)
### Start/End point definition
### Scaling and align videos
### Interpolating `none` frames

## Model development
### Landmark -> LSTM method
#### Overview
#### Feature Engineering
#### Data Augmentation
#### Models

## Results
### Overview
### Analysis
### Future ideas

## Demo app