[TOC]

# CS 230

## Introduction

* ANI - specific systems. AGI - building a generalized intelligence
* Part of a corpus of other AI tools - deep learning, probabilistic graphical models, planning, search, knowledge graphs, game theory. While most have made steady linear progress, deep learning sees a more exponential growth curve.

## C1 - Neural Networks and Deep Learning

### Module 1 - Intro to Deep Learning

* Housing price prediction - say you have 6 houses with data on size and price. This is a classic supervised learning problem, could fit a regression, might also have a kink at 0 to prohibit negative prices. One could think of this model as a simple neural network.
* Size x -> O -> price y. The intermediate step is a neuron - a function that transforms the input to an output
* We often see the 0 kinked function in NN - ReLU function (Rectified Linear Unit). Linear with pos slope above a certain value, zero below.
* Given additional features about a house: could take in size and # bedrooms  as inputs to family size node. Zip code is input into walkability node. Wealth and zip code inputs to school quality node. Finally family size, walkability, school quality are inputs to a node predicting price. Input and output is x(size, # bedrooms, zip code, wealth) and y(price), while the things in the middle are figured out by the network itself
* The hidden units take in all of the inputs in x - this means they are densely connected layers. 

##### Supervised Learning with Neural Networks

* Have input x and want to learn a function mapping to an output y
* Different NN types may be useful for different applications - real estate, advertising might use standard NN. Image recognition uses CNNs. Sequence data like speech recognition typically uses RNNs
* Structured data - databases of data. Unstructured data - raw audio, images, text. While often hear about NNs on unstructured, much of the economic value of NNs has been realized from structured data.

##### Recent Developments in Deep Learning

* Plot of amount of data vs. performance. While most learning methods plateau and cease to improve with a greater amount of data, NN’s continue to improve, and the deeper the network the more they continue to improve.
* Amount of labeled data has increased in recent times. We denote size of training set by $m$. 
* In regime of small training sets, the relative ordering of the learning methods is not well defined - not obvious which method is best. Only in large m do we see consistently large NN’s dominating. 
* Improvements in data size, computation, and algorithmic innovations. Many algorithmic improvements have allowed for faster training - such as switching from a sigmoid to a ReLU. The extremes of the sigmoid have very small derivative - gradient is small so learning is slow. ReLU has slope of 1 for all points above the kink - makes gradient descent much faster.
* Training a network is iterative - idea -> code -> experiment -> repeat. The learning cycle for the programmer becomes much tighter, allowing more creative and interactive modeling.

### Module 2 - Neural Networks Basics

##### Binary Classification

* Input -> 1 or 0. Binary values could stand for anything (cat or non cat). y is the output label
* Computer images stored in 3 RGB matrices. If image 64 x 64 pixels, have 3 x 64 x 64 pixel values. We unroll these pixel values into a feature vector x. $x = \left[\begin{array}{c}255\\231 \\ ... \end{array}\right]$. 64 x 64 x 3 = 12288. Say n = $n_X = 12288$
* $(x,y)$ is a single training example. $x \in \R^{n_x}, \; y \in \{0,1\}$
* m training examples: $\{(x^{(1)}y^{(1)}),...,(x^{(m)}y^{(m)})\}$. When we use $m=m_{train}$ vs $m_{test} =$ number of test examples.
* Define matrix $X = \left[\begin{array}{c}x^{(1)}&x^{(2)}&...&x^{(m)} \end{array}\right]$ with dimensions $n_x \times m$. Each $x^{(i)}$ is a column. X.shape = $n_x \times m$
* Define matrix $Y = \left[\begin{array}{c}y^{(1)}&y^{(2)}&...&y^{(m)} \end{array}\right]$ so $Y \in \R^{1 \times m}$. Y.shape = 1 x m

##### Logistic Regression

* Given x want $\hat{y} = P(y = 1 |x)$. We know $x \in \R^{n_x}$, then parameters: $w \in \R^{n_x}$ and $b \in \R$
* Could try output $\hat{y} = w^Tx + b$ - this is linear regression, but not great here because it does not force $0 \leq \hat{y} \leq 1$. 
* Instead use $\hat{y} = \sigma(w^Tx + b)$ - wrapped in the sigmoid function. Let $z = w^T + b$. Then $\sigma(z) = \frac{1}{1+e^{-z}}$. 
* If z is large then $\sigma(z) \approx 1$. If z is very small then $\sigma(z) \approx 0$
* We will handle w and b separately instead of folding into a new x vector ($\hat{y} = \sigma(\theta^Tx)$, inputing a $\theta_0$ as b)
* Loss Function
  * Given $\left\{\left(x^{(1)}, y^{(1)}\right), \ldots,\left(x^{(m)}, y^{(m)}\right)\right\}$ want $\hat{y}^{(i)} \approx y^{(i)}$. Note the superscript i, refers to data associated with the ith training example.
  * Loss to measure how good our estimate is compared to true value of y. We won’t use squared error here since it leads to local optima. 
  * Loss function: $L(\hat{y}, y) = -(ylog(\hat{y}) + (1-y)log(1 - \hat{y}))$. The loss function is a measure for a single training example.
  * If y = 1, $L(\hat{y}, y) = -log(\hat{y})$. We want $log(\hat{y})$ to be as large as possible, then want $\hat{y}$ to be large - since bounded above by 1, want it to be close to 1. 
  * If y = 0, $L(\hat{y}, y) = -log(1 - \hat{y}))$ - want $log(1-\hat{y})$ to be large, which means we want $\hat{y}$ as small as possible. Bounded below by 0 so we push it to be as close to 0 as possible.
* Cost Function
  * Loss over all training examples
  * $J(w,b) = \frac{1}{m}\sum_{i=1}^m L(\hat{y}^{(i)}, y^{(i)}) = -\frac{1}{m}\sum_{i=1}^m[y^{(i)}log(\hat{y}^{(i)}) + (1-y^{(i)})log(1 - \hat{y}^{(i)})]$

##### Gradient Descent + Derivatives

* Want to find w,b that minimize J(w,b). Graphically, we have planar axes w and b, and z-axis = J(w,b) - the cost function is a surface above w, b and want to find the value of w, b at the sink of the surface. J is a convex function, so we should be able to find a global minimum. 
* Initialize w and b to some values - almost any initialization is fine, but often use 0. Gradient descent starts at the initial point and takes a step in the direction of steepest descent. Iterate over and over until convergence around optimum is reached.
* Gradient Descent Procedures
  * Repeat: $w := w - \alpha \frac{\partial J(w,b)}{\partial w}$. 
    * Note $\alpha$ is the learning rate. Will define $dw := \frac{\partial J(w,b)}{\partial w}$. 
    * If w is too large, the slope will be positive and we subtract off a positive number making w smaller. If w is too small, the slope is negative and subtracting off the derivative will make w larger with each step.
  * In same loop, repeat $b := b - \alpha\frac{\partial J(w,b)}{\partial b}$
    *  Will define in code $db := \frac{\partial J(w,b)}{\partial b}$. 

##### Computation Graph

* J(a,b,c) = 3(a + bc)
* 3 distinct steps to compute this function. 
  * 1) u =bc 2) v = a + u 3) J = 3v
* Take these three steps and place them in a computation graph:

![Screen Shot 2020-01-07 at 5.11.06 PM](/Users/spencerbraun/Documents/Notes/Stanford/CS230/Screen Shot 2020-01-07 at 5.11.06 PM.png)

* For output variable J that we want to optimize, we compute the value of J from a left to right pass. To compute derivatives we will compute from right to left
* Taking derivatives - $\frac{dJ}{dv}$? Since J = 3v, v=11, a small change in v increases J by 3 times as much. $\frac{dJ}{dv} = 3$
* $\frac{dJ}{da}$ - J does not depend directly on a, so we rely on the chain rule. $\frac{dJ}{da} = \frac{dJ}{dv} \frac{dv}{da}$
  * Note for variables in code, use the notation $dvar := \frac{dFinalOutputVar}{dvar}$
* Continuing $\frac{dJ}{du} = \frac{dJ}{dv} \frac{dv}{du}$ - same as for a since at the same level. For the next level down $\frac{dJ}{db} = \frac{dJ}{du} \frac{du}{db}$ - we have already found $\frac{dJ}{du}$, so we can plug in our result from the prior step - at each step can just use a single chain rule. This is why it is most efficient to compute the derivatives from right to left.

##### Applying Gradient Descent to Logistic Regression

* For a single example using Loss:
  * $\hat{y} = a = \sigma(z)$
  * ![Screen Shot 2020-01-07 at 5.34.25 PM](/Users/spencerbraun/Documents/Notes/Stanford/CS230/Screen Shot 2020-01-07 at 5.34.25 PM.png)
  * $z = w_1x_1 + w_2x_2 + b \rightarrow \hat{y} = a = \sigma(z) \rightarrow L(a, y)$
  * $da = \frac{\partial L(a,y)}{\partial a} = -y/a + \frac{1-y}{1-a}$, $dz = \frac{\partial L(a,y)}{ \partial z} = \frac{\partial L}{\partial a}\frac{\partial a}{\partial z}= a -y$, $\frac{\partial L}{\partial w_1} = x_1 dz$, etc.
* For an entire training set using Cost:
  * Now using $J(w,b) = \frac{1}{m}\sum_{i=1}^m L(\hat{y}^{(i)}, y^{(i)})$ 
  * $\frac{\partial }{\partial w_1} J(w,b) = \frac{1}{m} \sum^m \frac{\partial}{\partial w_1} L(a^{(i)}, y^{(i)})$ - simply use the output from the previous step and average over all training examples.
  * Initialize J = 0, $dw_1 =0, \; dw_2 = 0,\; db=0$
  * For i=1 to m: 
    * $\begin{array}{l}
      {z^{(i)}=\omega^{T} x^{(i)}+b} \\
      {a^{(i)}=\sigma\left(z^{(i)}\right)} \\
      {J=\left[y^{(i)} \log a^{(i)}+\left(1-y^{(i)}\right) \log \left(1-a^{(i)}\right)\right]}\\ d z^{(i)}=a^{(i)}-y^{(i)}\\
      d w{1} +=x_{1}^{(i)} d z^{(i)}\\
      d w_{2} +=x_{2}^{(1)} d z^{(i)}\\ db += dz^{(i)}\end{array}$
  * Finally divide J, dw1, dw2, db by m since we are taking averages. At the end each accumulator variable equal to something like $dw_1 = \frac{\partial J}{ \partial w_1}$, etc.
  * Then $w_1 := w_1 - \alpha \times dw_1$, etc. This is a single step of gradient descent - we then need to repeat many times.

##### Vectorized Python

* Recall $z = w^Tx,\; w \in \R^n, \; x \in \R^n$

* `z = np.dot(w, x)+ b` computes $w^Tx + b$

* ```python
  import numpy as np
  
  # vectorized version
  a = np.random.rand(100000)
  b = np.random.rand(100000)
  c = np.dot(a,b)
  
  # loop version
  c = 0
  for i in range(100000):
   c += a[i] * b[i]
  ```

* GPU / CPU both have SIMD, single instance multiple data - vectorized computations allow for parallelization

* Whenever possible, avoid explicit for-loops

* Example: $u = Av$ then $u_i = \sum_jA_{ij}v_j$. Vectorized: `u = np.dot(A,v)`

* Example: vector v, want to exponentiate each value. `u = np.exp(v)`. Other useful element-wise functions: np.log, np.abs, np.maximum

* In logistic regression

  * Need to compute $z^{(1)}=w^{T} x^{(1)}+b, \;a^{(1)}=\sigma\left(z^{(1)}\right)$ for each data point
  * Construct 1 x m matrix $Z=\left[z^{(1)} z^{(1)} \cdots z^{(m)}\right]=\omega^{\top} X +\left[b b...b\right] = \left[\begin{array}{c} w^Tx^{(1)} + b & ... & w^Tx^{(m)} + b \end{array}\right]$
  * In python: `Z = np.dot(w.T, x) + b` - here python uses broadcasting, expanding b to a row vector of the constant repeated to the right dimensions. 
  * Similarly $A = \left[a^{(1)} ...a^{(m)}\right] = \sigma(Z)$

* Gradients

  * Define $dZ = \left[dz^{(1)} ...dz^{(m)}\right]$, $Y= \left[y^{(1)} ...y^{(m)}\right]$. Then $dZ = A - Y$
  * `db = 1/m * np.sum(dZ)`, `dw = 1/m * X * dZ.T` 

* Broadcasting

  * Say we have matrix 3 x 4. Want to sum for each columns and divide to get percents instead of gross. 

  * ```python
    import numpy as np
    A = np.array([
      [56., 0., 4.4, 68.],
      [1.2, 104., 52.0, 8.],
      [1.8, 135., 99.0, 0.9]])
    cal = A.sum(axis=0)
    #broadcasting - dividing 3x4 matrix by 1x4
    percentage = 100*A/cal.reshape(1,4)  #note - don't actually need reshape here
    ```

  * Python is autoexpanding during broadcasting - generally (m, n) # (1,n) -> (m,n);  (m, n) # (m,1) -> (m,n). Can also perform with single row / col vectors and scalars.
  * Note calling vectors with np can create a rank 1 array in python - neither column nor row. Specify actual dimensions in calls to get expected behavior, should see 2 square brackets not 1. Debugging tip: throw in `assert(a.shape == (5,1))` to check periodically the data looks as expected 

  

