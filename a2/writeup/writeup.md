---
title: CSC321 A2 ; classifying handwritten digits using single and muti-layer neural networks
author: Maxwell Huang-Hobbs (g4rbage)
---

# Part 1 - Description of Dataset

The dataset is the MNIST digit dataset, consisting of 1000 handwritten characters of each of [0,1,2,3,4,5,6,7,8,9] represented as 28x28 grayscale images. 

\begin{figure}[!h]
\centering
\captionsetup[subfigure]{labelformat=empty}
\begin{tabular}{cccccccccc}
\subfloat[]{\includegraphics[width = 0.25in]{part1/0_0.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/0_1.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/0_2.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/0_3.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/0_4.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/0_5.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/0_6.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/0_7.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/0_8.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/0_9.png}} \\
\subfloat[]{\includegraphics[width = 0.25in]{part1/1_0.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/1_1.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/1_2.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/1_3.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/1_4.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/1_5.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/1_6.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/1_7.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/1_8.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/1_9.png}} \\
\subfloat[]{\includegraphics[width = 0.25in]{part1/2_0.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/2_1.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/2_2.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/2_3.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/2_4.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/2_5.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/2_6.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/2_7.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/2_8.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/2_9.png}} \\
\subfloat[]{\includegraphics[width = 0.25in]{part1/3_0.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/3_1.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/3_2.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/3_3.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/3_4.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/3_5.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/3_6.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/3_7.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/3_8.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/3_9.png}} \\
\subfloat[]{\includegraphics[width = 0.25in]{part1/4_0.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/4_1.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/4_2.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/4_3.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/4_4.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/4_5.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/4_6.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/4_7.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/4_8.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/4_9.png}} \\
\subfloat[]{\includegraphics[width = 0.25in]{part1/5_0.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/5_1.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/5_2.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/5_3.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/5_4.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/5_5.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/5_6.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/5_7.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/5_8.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/5_9.png}} \\
\subfloat[]{\includegraphics[width = 0.25in]{part1/1_0.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/6_1.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/6_2.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/6_3.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/6_4.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/6_5.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/6_6.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/6_7.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/6_8.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/6_9.png}} \\
\subfloat[]{\includegraphics[width = 0.25in]{part1/7_0.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/7_1.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/7_2.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/7_3.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/7_4.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/7_5.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/7_6.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/7_7.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/7_8.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/7_9.png}} \\
\subfloat[]{\includegraphics[width = 0.25in]{part1/8_0.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/8_1.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/8_2.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/8_3.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/8_4.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/8_5.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/8_6.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/8_7.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/8_8.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/8_9.png}} \\
\subfloat[]{\includegraphics[width = 0.25in]{part1/9_0.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/9_1.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/9_2.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/9_3.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/9_4.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/9_5.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/9_6.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/9_7.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/9_8.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part1/9_9.png}} \\
\end{tabular}
\caption{Examples of input data}
\end{figure}

# Part 2 - Implementing the network

**Not: The first half of this assignment was implemented by concatenating biases to the front of the weight matrix, concatenating a row of constants to the front of the input matrix**

**Additionally, N was used as the first matrix index**

The network is implemented with the following function:

~~~ python
def linear_network(x, W):
    return softmax(dot(x, W))
~~~


# Part 3 - Gradient of the Single Layer Network

The gradient of the network was implemented using the following function

~~~ python
def grad_neg_log_likelihood(inp, weights, targets):
    N, _ = inp.shape
    I, O = weights.shape

    probability = softmax(dot(inp.reshape((N, I)), weights)

     = (output)
    pdiffs = (probability - targets)
    pdiffs = tile(pdiffs.reshape((N, 1, O)), (1, I, 1))
    inp_expanded = inp.reshape((N, I, 1))
    inp_expanded = tile(inp_expanded, (1, 1, O))

    return pdiffs * inp_expanded
~~~

\clearpage
\newpage

# Part 4 - Verifying the Gradient of the Single Layer network.

\begin{wrapfigure}[13]{r}{0.5\textwidth}
\vspace{-20pt}
\centering
\includegraphics[width=0.4\textwidth]{part4/difference_histogram.png}
\caption{ difference $(grad_{approx} - grad_{calc})$ between the approximate and calculated gradient for the linear network}
\end{wrapfigure}


The gradient of the network was approximated by changing the weight of each value in the weight matrix by a small step value (0.01) upwards and downwards, and calculating the slope of the cost over that difference 

The difference between corresponding values in the approximate and calculated gradients is highly centered around 0, which would suggest that the gradient function is accurate.

The large tails on either side of the distribution could be explained by the approximation function stepping over abrupt changes in the cost function. This could be corrected for by using a smaller step function for approximating the gradient.


# Part 5 - Gradient Descent with the Single Layer Network

\begin{figure}[!h]
\vspace{-20pt}
\centering
\includegraphics[width=0.8\textwidth]{part5/learning_rate.png}
\caption{Learning rate for the single layer neural network versus generations of the network}
\end{figure}

The neural network was trained on the training set with the gradient function from Part 4 using the batch size of 50 samples from each digit (500 total), and a constant learning rate of 0.01.

The final neural network performed with  $96.1$% accuracy on the testing set.

\begin{figure}[!h]
\centering
\captionsetup[subfigure]{labelformat=empty}
\begin{tabular}{cccccccccc}
\subfloat[]{\includegraphics[width = 0.25in]{part5/succ_0.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part5/succ_1.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part5/succ_2.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part5/succ_3.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part5/succ_4.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part5/succ_5.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part5/succ_6.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part5/succ_7.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part5/succ_8.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part5/succ_9.png}} \\
\subfloat[]{\includegraphics[width = 0.25in]{part5/succ_10.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part5/succ_11.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part5/succ_12.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part5/succ_13.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part5/succ_14.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part5/succ_15.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part5/succ_16.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part5/succ_17.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part5/succ_18.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part5/succ_19.png}} \\
\end{tabular}
\caption{Examples of images the network correctly classifies}
\end{figure}


\begin{figure}[!h]
\centering
\captionsetup[subfigure]{labelformat=empty}
\begin{tabular}{cccccccccc}
\subfloat[]{\includegraphics[width = 0.25in]{part5/failure_0.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part5/failure_1.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part5/failure_2.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part5/failure_3.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part5/failure_4.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part5/failure_5.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part5/failure_6.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part5/failure_7.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part5/failure_8.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part5/failure_9.png}} \\
\end{tabular}
\caption{Examples of images the network fails to  classify}
\end{figure}


The images the neural network failed to correctly classify are mostly either drawn with thin strokes, or skewed in some direction.

Failure to classify digits with various stroke widths can be explained by the fact that the neural network directly maps each intensity on the input layer to some output on the output layer, so a thinly stroked digit would have low intensity in many of the highly-weighted outputs.

Failure to classify skewed digits could likewise be explained by the structure of the neural network.

# Part 6 - Visualizing the weights of the inputs of the neural network
The weights of the inputs to the neural network look like blurred versions of their corresponding digits. They also seem to have some negative noise around the outside of the borders of each digit, which is likely weighting against the other digits.

\begin{figure}[!h]
\centering
\captionsetup[subfigure]{labelformat=empty}
\begin{tabular}{cccccccccc}
\subfloat[]{\includegraphics[width = 1in]{part6/part6_0.png}} &
\subfloat[]{\includegraphics[width = 1in]{part6/part6_1.png}} &
\subfloat[]{\includegraphics[width = 1in]{part6/part6_2.png}} &
\subfloat[]{\includegraphics[width = 1in]{part6/part6_3.png}} &
\subfloat[]{\includegraphics[width = 1in]{part6/part6_4.png}} \\
\subfloat[]{\includegraphics[width = 1in]{part6/part6_5.png}} &
\subfloat[]{\includegraphics[width = 1in]{part6/part6_6.png}} &
\subfloat[]{\includegraphics[width = 1in]{part6/part6_7.png}} &
\subfloat[]{\includegraphics[width = 1in]{part6/part6_8.png}} &
\subfloat[]{\includegraphics[width = 1in]{part6/part6_9.png}} \\
\end{tabular}
\caption{input weights of the neural network}
\end{figure}

\clearpage

# Part 7 - Implementation & Explanation of the Gradient Function

The following is the implementation of the gradient function used in this project. Some matrix reshaping at the beginning is omitted (accounting for the shape of the data)

~~~ python
def dtanh(y):
    return 1.0 - (y** 2)

def grad_multilayer((W0, b0), (W1, b1), inp, expected_output):
    # convenience variables
    I, N = inp.shape
    H, O = W1.shape

    # run through the neural network
    L0, L1, prediction = forward(inp, (W0, b0), (W1,b1))

    # partial derivatives 
    dCdL1 = (prediction - expected_output)
    dL1dL0 = dtanh(W1)
    dCdL0 = dot(dL1dL0, dCdL1)

    # derivative of layers
    dL1dW1 = dtanh(L0)  # L1 = tanh(dot(W1, L0))
    dL0dW0 = dtanh(inp) # L0 = tanh(dot(W0, inp)

    reshape and do a scalar multiplication instead of tiling and doing a dot product to avoid issues
    gradients_W1 = dCdL1.reshape((1, O, N)) * dL1dW1.reshape((H, 1, N))
    gradients_W0 = dCdL0.reshape((1, H, N)) * dL0dW0.reshape((I, 1, N))

    return (gradients_W0, dCdL0), (gradients_W1, dCdL1), prediction

~~~

### Partial Derivatives
1.
    ~~~
    dCdL1 = (prediction - expected_output)
    ~~~
    $\dfrac{\delta C}{\delta L_1} = \text{prediction} - \text{expected\_output}$
    given in the slides on one-hot encoding

2.
    ~~~
    dL1dL0 = dtanh(W1)
    ~~~
    Let $M$ be the output of the linear layer 
    $L_0 \rightarrow W_1 \rightarrow M$

    $\dfrac{\delta L_1^i}{\delta L_0^j} =
        \dfrac{\delta L_1^i}{\delta M}
        \dfrac{\delta M}{\delta L0}$

    $\dfrac{\delta M_1^i}{\delta L_0^j} M_1^i = 
        \dfrac{\delta M_1^i}{\delta L_0^j} \sum_{J} W_1^{i,J} L_0^j = 
        (W_1^{i,0} * 0 + .. + W_1^{i,j} + .. + W_1^{i,300} * 0) =
        W_1^{i,j}$

    $\dfrac{\delta L_1^i}{\delta M_1^i} =
        \dfrac{\delta}{\delta M_1^i} tanh(M_1^i) = dtanh(M_1^i)$

    Combining these partial derivatives we get,

    $\dfrac{\delta L_1^i}{\delta L_0^j} =
        dtanh(M_1^i) W_1^{i,j}$

    TODO FIX IN CODE

3. 
    ~~~
    dCdL0 = dot(dL1dL0, dCdL1)
    ~~~

    By the chain rule, and parts (1.) and (2.)

### Derivatives of Layers
1. 
    ~~~
    dL1dW1 = dtanh(L0)
    ~~~

    $L_1^i = tanh(\sum_J(W_1^{i,j}, L_0^j))$
    $\cfrac{\delta}{\delta W_1^{i,j}} L_1^i =
        \cfrac{\delta L_1^i}{\delta M^i}
        \cfrac{\delta M^i}{\delta W_1^{i,j}}$

    $\cfrac{\delta  L_1^i }{\delta W_1^{i,j}}=
        dtanh(M^i)
        \cfrac{\delta M^i}{\delta W_1^{i,j}}
            \sum_J W_1^{i,J} * L_0^J$

    $\cfrac{\delta  L_1^i }{\delta W_1^{i,j}}=
        dtanh(M^i)
        (0 * L_0^0 + ..  +L_0^j + .. + 0 * L_0^300)$

    $\cfrac{\delta L_1^i}{\delta W_1^{i,j}} =
        dtanh(M^i) L_0^j$

    TODO fix in code

2. 
    ~~~
    dL0dW0 = dtanh(inp)
    ~~~

    By the same process as (1.), 

    $\cfrac{\delta L_0^i}{\delta W_0^{i,j}} =
        dtanh(M) inp^j$

    TODO fix in code

3. 
    ~~~
    gradients_W1 = dCdL1.reshape((1, O, N)) * dL1dW1.reshape((H, 1, N))
    gradients_W0 = dCdL0.reshape((1, H, N)) * dL0dW0.reshape((I, 1, N))
    ~~~

    By the chain rule,
    $\cfrac{C}{\delta W_1} = \cfrac{\delta C}{\delta L_1} \cfrac{\delta L_1}{\delta W_1}$

    $\cfrac{C}{\delta W_0} = \cfrac{\delta C}{\delta L_0} \cfrac{\delta L_0}{\delta W_0}$

    A multiplication between matrices of mismatched dimensions broadcasts the arrays accross each other, which is equivalent to tiling and doing a dot product between the arrays

    TODO does this even work?

\clearpage

# Part 8 - Approximation to the Gradient

The difference between the actual and expected gradients was mostly promising (centered around 0, low spread). However, the implementation of the was likely inaccurate in some areas, as the distribution of the gradients of the offset matrix $b_1$ was unpatterned and widely spread.


\begin{figure}[!h]
\centering
\captionsetup[subfigure]{labelformat=empty}
\begin{tabular}{cc}
\subfloat[]{\includegraphics[height = 2in]{part8/difference_histogram_W0}} &
\subfloat[]{\includegraphics[height = 2in]{part8/difference_histogram_W1}} \\
\subfloat[]{\includegraphics[height = 2in]{part8/difference_histogram_b0}} &
\subfloat[]{\includegraphics[height = 2in]{part8/difference_histogram_b1}} \\
\end{tabular}
\caption{difference approximated - calculated for all values in the gradients of each of $W_0$ (top left), $W_1$ (top right), $b_0$(bottom left), and $b_1$(bottom right)}
\end{figure}


\begin{wrapfigure}[13]{r}{0.5\textwidth}
\vspace{-40pt}
\centering
\includegraphics[width=0.4\textwidth]{part9/multilayer_learning_curve.png}
\caption{ Learning curve of the multilayer neural network }
\end{wrapfigure}

# Part 9 - Training with the multilayer gradient 

As one would expect with a partially incorrect implementation of the gradient function, the neural network did not perform well. 

It hovers briefly around $44$% accuracy, before quickly falling off to around $10$% accuracy. This would suggest that the network is memorizing the features of only one of the classes of digits, and always reporting the digit to b that output

\clearpage

\begin{figure}[!h]
\centering
\captionsetup[subfigure]{labelformat=empty}
\begin{tabular}{cccccccccc}
\subfloat[]{\includegraphics[width = 0.25in]{part9/multilayer_succ_0.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part9/multilayer_succ_1.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part9/multilayer_succ_2.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part9/multilayer_succ_3.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part9/multilayer_succ_4.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part9/multilayer_succ_5.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part9/multilayer_succ_6.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part9/multilayer_succ_7.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part9/multilayer_succ_8.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part9/multilayer_succ_9.png}} \\
\subfloat[]{\includegraphics[width = 0.25in]{part9/multilayer_succ_10.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part9/multilayer_succ_11.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part9/multilayer_succ_12.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part9/multilayer_succ_13.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part9/multilayer_succ_14.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part9/multilayer_succ_15.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part9/multilayer_succ_16.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part9/multilayer_succ_17.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part9/multilayer_succ_18.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part9/multilayer_succ_19.png}} \\
\end{tabular}
\caption{Examples of images the network fails to  classify}
\end{figure}


\begin{figure}[!h]
\centering
\captionsetup[subfigure]{labelformat=empty}
\begin{tabular}{cccccccccc}
\subfloat[]{\includegraphics[width = 0.25in]{part9/multilayer_failure_0.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part9/multilayer_failure_1.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part9/multilayer_failure_2.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part9/multilayer_failure_3.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part9/multilayer_failure_4.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part9/multilayer_failure_5.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part9/multilayer_failure_6.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part9/multilayer_failure_7.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part9/multilayer_failure_8.png}} &
\subfloat[]{\includegraphics[width = 0.25in]{part9/multilayer_failure_9.png}} \\
\end{tabular}
\caption{Examples of images the network fails to  classify}
\end{figure}

# Part 10 - Visualizing the input layer of the Neural Network

The input layer to the multilayer neural network are mostly just random noise, though some of them appear to be fitting to parts of different digits

\begin{figure}[!h]
\centering
\captionsetup[subfigure]{labelformat=empty}
\includegraphics[width = 1.5in]{part10/112.png}
\caption{A slice of the first layer in the neural network}
\end{figure}

This slice of the input layer seems to be fitting to the character 2, and against the space immediately around 2

\begin{figure}[!h]
\centering
\captionsetup[subfigure]{labelformat=empty}
\includegraphics[width = 1.5in]{part10/114.png}
\caption{Another slice of the first layer in the neural network}
\end{figure}

This slice of the input layer seems to be fitting the areas guaranteed to be in the character 3, and against other areas of the image. The break in the layer is possibly because that stroke in the character 3 is often varied
