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

With the extracted videos, a cleaning step was applied. This task should solve and help with a few issues:

1. **Homographs:** Like any other language, there are some cases of homographs in Portuguese, that is, words that have the same spelling and different meanings, an example of homograph in English is the word "bat" that could either refer to the animal or the object used to hit the ball in a baseball game. Due to the structure of the data, these words would be registered with the same label but have different signs.
2. **Sign sinonyms:** Some words could have more than one sign, depending on the region or signer. Similarly to the homographs, these words would be registred under the same label but have different signs in the videos.
3. **Lack of video examples:** Even though there were videos for every registered word in the datasets, some of them had fewer videos than others. For the project to be successful, it was necessary to have at least a certain amount of example videos to train the models.
4. **Healthcare related words:** The project's main goal was to act on translating LIBRAS specifically in healthcare settings. Therefore, it was necessary to select health related words to compose a target dataset.

Below, you can find the approach used to solve each of the issues above.

#### Video reviewing
[comment]: <TODO: continue video reviewing and data cleaning steps from here>


### Review steps
### Final dataset

## Preprocessing
### EDA
*DRAFT*
- Dataset differences
    - dimensions
        - grid of data sources & dims
    - frame rates
        - on same grid as above?
    - durations
        - plot for each label

*DRAFT*
### Pose estimation with [MediaPipe Holistic](https://ai.google.dev/edge/mediapipe/solutions/guide)
why:
    - will be used in preprocessing for motion detection, offset & scaling
    - Will be used as the base features that will be input to the model

### Start/End point definition
*DRAFT*
#### motion detection 
- we are explored various different methods for measuring motion between frames:
    - bf sub
    - Basic
    - Landmarks
- Each normalised and moving avgd for smooth results, and consistency between datasets
    - we are mainly interested in peaks, so normalising makes sense
    - We don’t have some sense of the correct ’absolute value’ across datasets
- Show some before & after
- show some example via gifs
- for some previous versions, we used a combination of multiple, but in the final version settled on just landmarks
#### detecting start and end
- basic method is just taking an absolute threshold for the start & end
- complex method involves 
    - detecting the first and last peak to get the correct general location 
    - taking the rate of change of the motion 
    - Search back/forward from the peaks for the index where the slope has an inflection point 
        - this should show when the movement really started 
    - Go 0.X seconds before this as a buffer
        - seconds rather than frames to be robust to different datasets
- in the final version we just went for the simple version 
    - both have threshold params that need tuned 
    - But simple has less and is more intuitive 
    - Without annotating our ground truth desired start / end, it is hard to tune this
        - all we could do was manually experiment 
- our method for tuning was just
    - run manual experiments for a bunch of thresholds
    - compare the variance of the start/end %
        - assumption is it should be less if there are less extreme cases
    - inspect the results of a sample of the data wit visualisation like these:
- from our limited time experimenting, complex was not significantly better than simple. 
    - with more time / annotations we could tune complex to be significantly better
    - Simple was also robust to jittery motion
    - We would need some more development to deal with this in complex 
    - But in this limited time, just go for simple
*DRAFT*
### Scaling and align videos

### Interpolating `none` frames
*DRAFT*
Context
- Format of MediaPipe output
    - For a frame, individual landmarks can't be none. Only the full group of landmarks can be none
    - There are a few reasons to be None, and they also depend on the type
- 99% of the time we have Nones, they are hand landmarks being None
    - This is because of how they are detected
    - A significant proportion of these are justified, the hand is not in the frame at the beginning or end
        - Ignoring sequences of Nones at the start / end, we still have quite a lot of Nones
        - (Plot showing None sequences)
    - A significant proportion of these problematic Nones are from the lowest resolution dataset, INES 

*DRAFT*

## Model development
### Landmark -> LSTM method
#### Overview
#### Train / Validation / Test split
#### Feature Engineering
#### Data Augmentation
#### Models

## Results
### Overview
### Analysis
### Future ideas

## Demo app