---  
title: CSC321 Assignment 1 
subtitle: Classifying Faces with K-Nearest Neighbors
author: 
    - Maxwell Huang-Hobbs (g4rbage)
date: Due Feb 3 2016
graphics: true
# toc: true
---  

# Overview

In this assignment, a k-nearest-neighbor classifier was trained on the faces of various actors and actresses. It was then used to classify new images of the same actors people by name and gender, and to classify the genders of actors and actresses not seen in the training set.

The classifier performed with a %63.3%% accuracy at recognizing known actors, $88.3$% accuracy at recognizing the genders of unknown actors, and $68.3$% accuracy at recognizing the genders of unknown actors.

# 1. Description of Dataset

The dataset consists of a series of photographs of 6 celebrities (Angie Harmon, Daniel Radcliffe, Gerard Butler, Lorraine Bracoo, Michael Vartan, and Perli Gilpin), with corresponding bounding boxes around the face of each individual.


\begin{figure}[h!]
\begin{centering}
    \begin{multicols}{3}
    \includegraphics[width=0.3\textwidth]{angie-harmon-perfect.jpg}
    \caption{an ideal sample}
    \vfill
    \includegraphics[width=0.3\textwidth]{gerard-butler-noncentered.jpg}
    \caption{a non-ideal sample: the bounding box is offset from the center
             of the face}
    \vfill
    \includegraphics[width=0.3\textwidth]{daniel-radcliffe-angled.jpg}
    \caption{a non-ideal sample: the face is at an angle and the bounding
             box captures the background}
    \end{multicols}
\end{centering}
\end{figure}

However generally the faces can be aligned with one another. The performance of the classifier could likely be improved by manually curating a dataset with similar face crops.

Images were collected with `get_data.py`.



## 2. Division of processed images
120 images of each subject were selected at random from the dataset, and 
then were divided randomly into the 'training', 'validation', and 'test' sets. (data in section 3.), with 100 in 'training', and 10 in each 'validation' and 'test'

Images were divided randomly with `create_samples.py`.



# 3. K-nearest-neighbors classification

## Finding the best value for k

When run on the validation set for values of k in the range $[1 .. 20]$, the best value for $k$ was found to be $k=2$ on the validation set, with a $62$% accuracy for recognizing an actor from $act$.



## Performance on the test set

When this value of k was used on the test set, it was correct
$60.00$% of the time. Below are cases where the classifier selected the wrong face.


input person       input face         nearest person   nearest neighbor  
------------       -----------------  --------------   -----------------
Gerard Butler      ![][person_real1]  Michael Vartan   ![][person_reco1]
Daniel Radcliffe   ![][person_real2]  Michael Vartan   ![][person_reco2]
Angie Harmon       ![][person_real3]  Gerard Butler    ![][person_reco3]
Angie Harmon       ![][person_real4]  Lorraine Bracco  ![][person_reco4]
Peri Gilpin        ![][person_real5]  Lorraine Bracco  ![][person_reco5]

[person_real1]: assets/person_mismatch/real01.jpg
[person_real2]: assets/person_mismatch/real02.jpg
[person_real3]: assets/person_mismatch/real03.jpg
[person_real4]: assets/person_mismatch/real04.jpg
[person_real5]: assets/person_mismatch/real05.jpg

[person_reco1]: assets/person_mismatch/recognized01.jpg
[person_reco2]: assets/person_mismatch/recognized02.jpg
[person_reco3]: assets/person_mismatch/recognized03.jpg
[person_reco4]: assets/person_mismatch/recognized04.jpg
[person_reco5]: assets/person_mismatch/recognized05.jpg

In general, the mis-recognized nearest neighbors share some attribute with the
input face other than the the person's face, e.g similar lighting, position of the face in the picture, or framing with bangs.



# 4. Graph of performance versus k

Accuracy appears to drop off quickly as k increases. This could be explained if the input dataset were broken into different areas based on non-face features of the picture (overall light  values in the picture, etc). This would mean that locally knn would perform well, but as k increased more faces from the same 'cluster' would be considered.

\begin{center}
\begin{tikzpicture}[trim axis left, trim axis left]
\begin{axis}[
    xtick=data,
    xlabel=k (samples),
    ylabel=accuracy(percent),
    enlargelimits=0.1,
    ybar interval=0.1,
]
\addplot[scatter, only marks] 
    coordinates {
        (1, 61.67)
        (2, 63.33)
        (3, 58.33)
        (4, 51.67)
        (5, 48.33)
        (6, 50.00)
        (7, 51.67)
        (8, 48.33)
        (9, 50.00)
        (10, 48.33)
        (11, 46.67)
        (12, 46.67)
        (13, 41.67)
        (14, 38.33)
        (15, 38.33)
        (16, 36.67)
        (17, 35.00)
        (18, 35.00)
        (19, 35.00)
        (20, 33.33)
    };
\end{axis}
\end{tikzpicture}
\end{center}

If this were the case, one solution would be to normalize the light levels in pictures. It could also be solved by clustering faces by their location in the face-space and performing k-means within each of those clusters. In general, knn will have these issues though, as without significant preprocessing, it has no way to distinguish a meaningful attribute of the input set.


k         person accuracy  
----     -----------------
01       61.67
02       63.33
03       58.33
04       51.67
05       48.33
06       50.00
07       51.67
08       48.33
09       50.00
10       48.33
11       46.67
12       46.67
13       41.67
14       38.33
15       38.33
16       36.67
17       35.00
18       35.00
19       35.00
20       33.33

\clearpage
\newpage

# 5. Performance of knn classifier on gender

The best k found for classifying by gender was also $k=2$, with an accuracy of $88.3$%.

\begin{center}
\begin{tikzpicture}[trim axis left, trim axis left]
\begin{axis}[
    xtick=data,
    xlabel=k (samples),
    ylabel=accuracy(percent),
    enlargelimits=0.1,
    ybar interval=0.1,
]
\addplot[scatter, only marks] 
    coordinates {
        (1,  86.67)
        (2,  88.33)
        (3,  88.33)
        (4,  85.00)
        (5,  85.00)
        (6,  85.00)
        (7,  88.33)
        (8,  86.67)
        (9,  88.33)
        (10, 85.00)
        (11, 86.67)
        (12, 86.67)
        (13, 81.67)
        (14, 80.00)
        (15, 80.00)
        (16, 76.67)
        (17, 78.33)
        (18, 78.33)
        (19, 75.00)
        (20, 73.33)
    };
\end{axis}
\end{tikzpicture}
\end{center}


k        gender accuracy    
----    ----------------- 
01      86.67%            
02      88.33%            
03      88.33%            
04      85.00%            
05      85.00%            
06      85.00%            
07      88.33%            
08      86.67%            
09      88.33%            
10      85.00%            
11      86.67%            
12      86.67%            
13      81.67%            
14      80.00%            
15      80.00%            
16      76.67%            
17      78.33%            
18      78.33%            
19      75.00%            
20      73.33%          

This increase in performance is to be expected, since the knn classifier must be _at least_ as good at recognizing a gender as it is at recognizing an actor / actress from the training set.

# 6. Performance on gender of actors not in the training set.

In order to determine the performance of the classifier on the faces of people not in the training set, 10 images of Julia Louis-Dreyfus, Dana Delany, Holly Marie Combs, Cary Elwes, Chris Klein, and Andy Richter were tested with the gender classifier.

When tested against actors / actresses not in *act*, the knn gender classifier performs with $68.3$% accuracy. This is notably lower than the performance on the same actors / actresses in *act*.

It is likely the case that the numbers from part 4 are artificially inflated - because we are working on the same people as the test set, the gender classifier can rely on similarities between input images and the training set that are not necessarily tied to a person's gender.

The actors / actresses used in the test set for this part can be found in the attached `different_actors_dataset.json`

