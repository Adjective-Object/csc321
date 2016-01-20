---  
title: CSC321 Assignment 1
author: Maxwell Huang-Hobbs
date: Due Feb 3 2016
bibliography: writeup-library.bib
csl: chicago.csl

graphics: true
---  

# 1. Description of Sampleset

The dataset consists of a series of photographs of 6 celebrities (Angie Harmon, Daniel Radcliffe, Gerard Butler, Lorraine Bracoo, Michael Vartan, and Perli Gilpin), with corresponding bounding boxes around the face of each individual.

\begin{figure}[h!]
\begin{centering}
    \begin{multicols}{3}
    \includegraphics{angie-harmon-perfect.jpg}
    \caption{an ideal sample}
    \vfill
    \includegraphics{gerard-butler-noncentered.jpg}
    \caption{a non-ideal sample: the bounding box is offset from the center
             of the face}
    \vfill
    \includegraphics{daniel-radcliffe-angled.jpg}
    \caption{a non-ideal sample: the face is at an angle and the bounding
             box captures the background}
    \end{multicols}
\end{centering}
\end{figure}

Images were collected with `get_data.py`.

## 2. Division of processed images
120 images of each subject were selected at random from the dataset, and 
then were dividedr andomly into the 'training', 'validation', and 'test' sets.

Images were divided with `create_samples.py`.

## 3. K-nearest-neighbors classification

# References

pandoc will put references from you library here.

