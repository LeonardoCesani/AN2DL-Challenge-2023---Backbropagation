
# Deep Learning Models for Image Classification and Time Series Forecasting

## Team: Backbropagation
- **Gianluca Canzii**: gianluca.canzii@mail.polimi.it
- **Leonardo Cesani**: leonardo.cesani@mail.polimi.it
- **Matteo Colella**: matteo1.colella@mail.polimi.it
- **Leonardo Spinello**: leonardo.spinello@mail.polimi.it

---

## Project Overview

This repository contains the scripts used to solve two deep learning problems: 
1. **Health-State Classification of Leaves**: A binary image classification task aimed at distinguishing between healthy and unhealthy leaves using transfer learning and fine-tuning techniques.
2. **Time Series Forecasting with Attention and LSTM**: A forecasting model built to predict future values of multiple time series using Long Short-Term Memory (LSTM) networks enhanced by attention mechanisms.

---

## Problem 1: Health-State Classification of Leaves

### Dataset
The dataset consists of **5200 RGB images** (96×96 pixels), split between 3199 "healthy" and 2001 "unhealthy" leaves. The goal is to build a model to classify these images accurately.

### Approach
1. **Preprocessing**: 
   - A custom script was developed to detect and remove outliers and duplicates from the dataset. This involved analyzing pixel imbalances in the RGB channels (specifically the red and blue channels in relation to the green channel).
   - Images with significant imbalances or that did not match the task specifications were removed from the dataset.

2. **Model Selection**:
   - Initial experiments involved building a quasi-VGG model with three convolutional layers, but it underperformed.
   - The team then moved on to **transfer learning**, selecting models like **EfficientNetV2S** and **ConvNeXtLarge**, which showed improved performance.

3. **Data Augmentation**:
   - To overcome the small dataset size and imbalance between the healthy/unhealthy classes, various data augmentation techniques were used.
   - A **class-wise data augmentation** strategy was applied using transformations like flipping, rotating, and modifying color channels. Bayesian optimization was employed to find the best augmentation parameters for each class.

4. **Ensemble Learning**:
   - Two separate models were developed (referred to as Model 1 and Model 2). These were then combined using ensemble learning, improving overall accuracy to **92%** on the test set.
   - The ensemble model aggregates predictions from both models, leading to better generalization and robustness.

### Key Files (`Classification` folder)
- `model_1.ipynb.py` and `model_2.ipynb.py`: Contains the code to implement the models.
- `AugmentedDatasetGenerator.py`: Contains the code to perform dataset augmentation using techniques like flipping, rotating, and color manipulation.
- `AugmentationParameterOptimization.py`: Implements class-wise augmentation and parameter tuning using Bayesian optimization.
- `ensemble_model.py`: Combines predictions from multiple models to improve accuracy.

### Final Results
The ensemble model obtained an accuracy of **92%** on the test set and achieved a **ranking of 14th** in the competition.

---

## Problem 2: Time Series Forecasting with Attention Mechanism

### Dataset
The dataset consists of **48,000 labeled time series**, each representing a sequence from different categories. The objective is to predict the next 18 time steps for each sequence.

### Approach
1. **Preprocessing**:
   - The time series dataset was split into subsequences, with each subsequence consisting of 209 elements.
   - Subsequences shorter than 209 elements were either padded or cropped based on their length. This was done after analyzing the impact of different threshold lengths on the dataset.

2. **Model Architecture**:
   - The main model architecture used for the forecasting task is based on the **Dual-Stage Attention-Based Recurrent Neural Network (DA-RNN)**, which utilizes both an encoder-decoder structure and a dual attention mechanism.
   - Two attention mechanisms were explored:
     - **Squeeze and Excitation (SE) Mechanism**: This mechanism adjusts the importance of different feature channels by using a combination of spatial squeezing and excitation to scale the channel-wise features.
     - **Convolutional Block Attention Module (CBAM)**: This mechanism focuses on both spatial and channel-wise attention to highlight important features and areas in the sequence.
   
3. **Category Information**:
   - To enhance model performance, categorical information was encoded and fed into the model alongside the sequence data. This enabled the network to use both sequence features and class-specific information for better forecasting.

4. **Ensemble Learning**:
   - Two models were built: one using SE attention and another using CBAM attention. An ensemble model combined predictions from both models, weighted by their performance on different categories.
   
5. **Autoregression**:
   - To handle the prediction of longer time horizons (18 steps), an autoregressive method was employed. After predicting the first set of values, these predictions were appended to the input to predict the next set of values iteratively.

### Key Files (`Forecasting` folder)
- `data_exploration.py`: Script for exploring, preprocessing, and splitting the dataset into subsequences.
- `models.py`: Contains the implementations of the SE and CBAM attention modules used in the DA-RNN architecture.

### Final Results
- The final ensemble model achieved a **Mean Squared Error (MSE) of 0.00746** and **ranked 1st** in the competition. 
- The combination of dual-stage attention, categorical input, and autoregressive forecasting contributed to the superior performance of the model.

## References

- [EfficientNetV2S](https://arxiv.org/abs/2104.00298)
- [ConvNeXtLarge](https://arxiv.org/abs/2201.03545)
- [DA-RNN](https://arxiv.org/abs/1709.01507)
- [SE Attention](https://arxiv.org/abs/1709.01507)
- [CBAM Attention](https://arxiv.org/abs/1807.06521)

---
