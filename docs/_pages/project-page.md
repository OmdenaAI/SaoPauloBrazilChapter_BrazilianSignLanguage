---
layout: post
title: Project Page
permalink: /
author: Ben Thompson
---
# **AI Sign Language Translation for Brazilian Healthcare - Omdena Project**

## **Introduction**

This project was done as part of an Omdena local chapter challenge with the São Paulo chapter. These challenges are **open source collaborations** between volunteer developers around the world, working to **create AI applications that solve social issues** for local communities. 

Brazil has a significant deaf community, with **Brazilian Sign Language (Libras)** recognized as an official language. However, **communication barriers** often exist in healthcare settings, potentially compromising the quality of care for deaf patients. This potentially lead to misdiagnoses, treatment errors, or a compromised patient experience.

This project aimed to address this by **developing a sign language translation model for Brazilian Sign Language (LIBRAS)**, and a demo web app to showcase the model.


This page contains an overview of the project work, and links to the other deliverables. You can navigate to the various sections of this "Project Page", and the more detailed "Report Page" using the side bar on the left.

## **Other pages**
The deliverables for this project consisted of a live demo application, the open source code, and a report of our work and results.

**Demo Application:**
[You can see the model in action and test it on your own videos in the **Demo App**](https://sao-paulo-brazil-chapter-brazilian-sign-language.vercel.app/).

<div align="center">
<img src="assets/demo_app_video.gif" alt="isolated" title="hover hint" style="width: 75%; border: 2px solid #ddd;"/>
<p><em>GIF showing the demo application in action</em></p>
</div>

**Open Source Code:**
[You can find the project code and instructions for setup/usage in our  **Github Repo**](https://github.com/OmdenaAI/SaoPauloBrazilChapter_BrazilianSignLanguage)

**Detailed Report:**
[This page has an overview of the project. For more details, jump to the **Report Page**](report)



# **Domain - Sign Language Processing (SLP)**


## **What is SLP?**

Sign Language Processing (SLP) is an AI field that combines Natural Language Processing and Computer Vision to process and analyze sign language content. 

Unlike most spoken languages sign languages lack a standard written representation, and they use a combination of hand movements, facial expressions, and body gestures to communicate meaning. These characteristics, and other details about the lingusitics of sign languages make SLP a very interesting, but also challenging field.

There are a variety of SLP tasks, like sign language recognition, sign language translation, sign language production, sign language detection, and sign language segmentation. In this project we focused on **sign language recognition**.

## **Approaches to SLP**

The best approach for modelling sign language depends on the specific task, and has evolved over time as the field advanced, and incorporated the latest advances in Natural Language Processing and Computer Vision.

<!-- Common techniques used for data representation, are processing raw video directly, pose estimation using tools like MediaPipe or OpenPose, and symbolic notation with 'glosses'. For feature extraction,  CNNs, pose landmark analysis, and motion tracking between frames are often used. For model architecture, LSTMs/RNNs, Transformers, and graph networks are used when modelling sequence data. -->

In this project, our approach was **pose estimation for feature extraction, modelled with RNN, LSTM, and Transformer models**.

## **Recommended Reads** 

For a comprehensive and up-to-date resource covering the field of SLP research and the dataset landscape, see [research.sign.mt](https://research.sign.mt/)

For a recent paper from covering the current state of the art in SLP, data availability, and the challenges of the field, see [A review of deep learning-based approaches to sign language processing - 2024](https://www.tandfonline.com/doi/full/10.1080/01691864.2024.2442721)

# **Project Summary**

## **Goal**

To address the problem of communication barriers in healthcare settings for the deaf community in Brazil, we would develop a sign language translation model for Brazilian Sign Language (LIBRAS). The deliverable will be a report of our work and results, and a demo web app to showcase the model & make it accessible to the public.

## **Results**

We used a combination of 4 public LIBRAS datasets to create a target dataset of 25 words related to the healthcare domain, with 6 videos each.

We developed a detailed preprocessing pipeline to standardize the conditions between each data source, and a training pipeline that allowed us to experiment with different feature engineering, data augmentation, model architectures and training strategies.

With 6 videos per word, for each word we used 4 videos for training, 1 for validation, and 1 for testing. We had an even distribution of data sources in each split to avoid bias. 

### **Details about the best performing model:**
- **LSTM architecture:**
  - 2 layers
  - 256 hidden units
- **20 frames per sample:**
  - Randomly sampled from the preprocessed frame series
- **189 input features per frame:**
  - Dropped the 150 landmark position coordinates
  - Engineered 181 features capturing distances & angles between landmarks, and movements  across consecutive frames
  - Additional 8 features representing various metadata
- **Data augmentation:**
  - Rotation (±10 degrees, applied to an entire series)
  - Gaussian Noise on Landmark coordinates (0.05 std, applied to each landmark on each frame)

### **Best model performance:**

<!-- | Evaluation Set         | Accuracy  | top-2 Accuracy  | top-3 Accuracy  | top-4 Accuracy  | top-5 Accuracy  | Loss  |
|----------------|--------|--------|--------|--------|--------|--------|
| Validation Set  | 66.37%   | 77.88%   | 84.07%   | 89.38%   | 93.81%   | 0.01231   |
| Test Set  | 66.37%   | 77.88%   | 84.07%   | 89.38%   | 93.81%   | 2.66401   | -->

<div style="width: 65%; margin: 0 auto;">
<table>
  <thead>
    <tr>
      <th style="width: 20%; text-align: left;">Metric</th>
      <th style="width: 45%; text-align: center;">Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Accuracy</td>
      <td style="text-align: center;">66.37%</td>
    </tr>
    <tr>
      <td>Top-2 Accuracy</td>
      <td style="text-align: center;">77.88%</td>
    </tr>
    <tr>
      <td>Top-3 Accuracy</td>
      <td style="text-align: center;">84.07%</td>
    </tr>
    <tr>
      <td>Top-4 Accuracy</td>
      <td style="text-align: center;">89.38%</td>
    </tr>
    <tr>
      <td>Top-5 Accuracy</td>
      <td style="text-align: center;">93.81%</td>
    </tr>
    <tr>
      <td>Loss</td>
      <td style="text-align: center;">2.66401</td>
    </tr>
  </tbody>
</table>
</div>

The training loss and validation loss for this model were `0.02336` and `0.00660` respectively. These are significantly lower than the test loss, **indicating overfitting**. 

We took **measures to prevent overfitting**: like randomly sampling frames, applying data augmentation, and stratifying the data sources in each split. The data augmentation is the reason the training loss is higher than the validation loss. However the model still overfits in the end. 

We expect that the small dataset size is a large reason the model overfits. To remedy this, a larger dataset would of course help, but in lieu of that, we could apply **more aggresive data augmentation** to help the model generalize better to new features in unseen data.


## **Contributors**

The main work for this project took place over 4 months, from February to June 2025. Below is a list of the people that contributed to the project. Feel free to reach out to them if you have questions about any aspect of the project. Some members have also made additional changes & improvements since the end of the main project period.

### **Project Leader**
<p><a href="https://www.linkedin.com/in/ben-d-thompson/"><strong>Ben Thompson</strong></a></p>
<ul>
<li>Tasks: Research Resources, Data Scraping, Data Preprocessing, Model Development, Demo App Development</li>
<li>Omdena Role: Project Leader & Lead ML Engineer</li>
</ul>

### **Task Leaders**

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;">

<div>

<p><a href="https://www.linkedin.com/in/ayushya-pare/"><strong>Ayushya Pare</strong></a></p>
<ul>
<li>Tasks: Research Resources, Data Scraping</li>
<li>Omdena Role: Lead ML Engineer</li>
</ul>

<p><a href="https://www.linkedin.com/in/gustavopsantos"><strong>Gustavo Santos</strong></a></p>
<ul>
<li>Tasks: Data Scraping, Data Cleaning & Organisation</li>
<li>Omdena Role: Lead ML Engineer</li>
</ul>

</div>

<div>

<p><a href="https://www.linkedin.com/in/anastasiia-derzhanskaia/"><strong>Anastasiia Derzhanskaia</strong></a></p>
<ul>
<li>Tasks: Model Development</li>
<li>Omdena Role: Lead ML Engineer</li>
</ul>

<p><a href="https://www.linkedin.com/in/patrick-fitz-b2186a11b/"><strong>Patrick Fitz</strong></a></p>
<ul>
<li>Tasks: Demo App Development</li>
<li>Omdena Role: Lead ML Engineer</li>
</ul>

</div>

</div>

### **Task Contributors**

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;">

<div>

<p><a href="https://www.linkedin.com/in/michael-spurgeon-jr-ab3661321/"><strong>Michael Spurgeon Jr</strong></a></p>
<ul>
<li>Tasks: Data Scraping, Model Development</li>
<li>Omdena Role: ML Engineer</li>
</ul>

<p><a href="https://github.com/prajinip13"><strong>Pooja Prasad</strong></a></p>
<ul>
<li>Tasks: Data Review & Cleaning</li>
<li>Omdena Role: Junior ML Engineer</li>
</ul>

<p><a href="https://www.linkedin.com/in/ethel-phiri261312/"><strong>Ethel Phiri</strong></a></p>
<ul>
<li>Tasks: Data Scraping</li>
<li>Omdena Role: Junior ML Engineer</li>
</ul>

<p><a href="https://www.linkedin.com/in/damla-helvaci/"><strong>Damla Helvaci</strong></a></p>
<ul>
<li>Tasks: Data Review & Cleaning</li>
<li>Omdena Role: Junior ML Engineer</li>
</ul>

</div>

<div>

<p><a href="https://www.linkedin.com/in/wafa-basoodan-9447616a/"><strong>Wafa Basoodan</strong></a></p>
<ul>
<li>Tasks: Model Development</li>
<li>Omdena Role: ML Engineer</li>
</ul>

<p><a href="https://www.linkedin.com/in/kunal-sood-3b993225b/"><strong>Kunal Sood</strong></a></p>
<ul>
<li>Tasks: Model Development</li>
<li>Omdena Role: Junior ML Engineer</li>
</ul>

<p><a href="https://github.com/guluzar-gb"><strong>Guluzar GB</strong></a></p>
<ul>
<li>Tasks: Data Review & Cleaning</li>
<li>Omdena Role: Junior ML Engineer</li>
</ul>

</div>

</div>

## **Project Tasks**

<!-- - Scraped data from 4 different sources
- Cleaned the data to create a dataset of 1000+ videos of LIBRAS signs
- Reviewed the data to find words with the most videos
- Created a target dataset of 25 words with high quality videos, and words related to the healthcare domain
- Developed an advanced preprocessing pipeline that allowed us to use data from multiple sources
- Developed a modelling code base that allowed us to experiment with different data sampling, data augmentation, model architectures and training strategies
- Ran a range of model experiments, and found the best performing model
- Developed a demo web app to showcase the model
- Created a report summarizing the project and the results -->

The project work was divided into separate tasks, with some of them able to be done in parallel. A task leader was assigned to each task, who would lead the work, coordinate with task contributors, and align with the project leader weekly about progress and direction.

### **1. Research & Planning**

To decide on the scope of the project and our plan for development, we conducted research and shared findings with each other. We investigated the SLP domain generally, the datasets available for LIBRAS, and the existing literature about SLP for LIBRAS.

### **2. Data Collection**

We used Selenium and BeautifulSoup to scrape videos and metadata from 4 different public sources to get a large dataset of ~8500 videos corresponding to ~2100 labels. After cleaning and excluding labels with less than 5 videos, we had 170 labels that were candidates for our target dataset. After reviewing the videos, and removing some labels that were not relevant to the healthcare domain, we had our target dataset of 25 words, with 6 videos per word.

<div align="center">
<img src="assets/all_signs_for_banana.gif" alt="isolated" title="hover hint" style="width: 100%; border: 2px solid #ddd;"/>
<p><em>GIF showing all sign videos in the dataset for the word 'banana'</em></p>
</div>

### **3. Data Preprocessing**

The videos from each data source had a lot of variety in their format (framerate, dimensions, etc.), and the conditions of the recorded signs (position of the signer, lighting, scale, etc.). We used Google's open source model MediaPipe Holistic to get pose estimation landmarks. 

<div align="center">
<img src="assets/pose_for_casa_on_orig_and_black_raw.gif" alt="isolated" title="hover hint" style="width: 85%; border: 2px solid #ddd;"/>
<p><em>GIF showing raw pose estimation landmarks for the sign 'casa'</em></p>
</div>

We used developed a detailed preprocessing pipeline with OpenCV and NumPy to standardize the conditions between each data source. The pipeline would trim the series to the start and end of the sign, adjust signers to the same scale,  align the signers to the same central position, and apply interpolation to fill in frames where landmark detection failed. 

<div align="center">
<img src="assets/pose_for_casa_on_black_preproc.gif" alt="isolated" title="hover hint" style="width: 85%; border: 2px solid #ddd;"/>
<p><em>GIF showing pose estimation landmarks before and after preprocessing for the sign 'casa'</em></p>
</div>


### **4. Model Development**

We used PyTorch to code the models (RNN, LSTM & Transformer) and training pipeline. The training pipeline was engineered to be platform agnostic, and resumable since we were collaborating internationally. We used Hydra & Tensorboard for managing and tracking model experiments. We used Google Colab for collaboratively training and testing models on both GPU and CPU

### **5. Report**

The first deliverable for this project was a report of our work and results. We used Seaborn and Matplotlib to create all the plots, visualizations, and gifs in the report. We used Jekyll and Github Pages to host the report as a static website.

### **6. Demo App**

The second deliverable for this project was a [**demo application**](https://sao-paulo-brazil-chapter-brazilian-sign-language.vercel.app/). Users can select from sample videos, or upload their own videos, and see the model's prediction & pose estimation result. We developed the Backend with Python & FastAPI, and deployed it with Hugging Face Spaces. We developed the Frontend with Next.js & React, and deployed it with Vercel.
