[TOC]

# Class

## Introduction

* ANI - specific systems. AGI - building a generalized intelligence
* Part of a corpus of other AI tools - deep learning, probabilistic graphical models, planning, search, knowledge graphs, game theory. While most have made steady linear progress, deep learning sees a more exponential growth curve.

## Practical Approaches to Deep Learning

* Taking the cat logistic classifier, if we wanted to classify multiple animals
  * Could have 3 output neurons instead of 1 - the neurons would be independent of each other. Then we would also need to change our input, now need labeled data that isn’t just cat / non-cat but labeled with the outputs we want from the new model.
  * One-hot encoding scheme - each index is responsible for an animal $\left[\begin{array}{c}0\\1\\0\end{array}\right] = \left[\begin{array}{c}cat\\dog\\giraffe\end{array}\right]$. If an image that has both cats and dogs, could have multiple indices turned on - multi hot encoding
  * Have to consider is our architecture compatible with multi-hot? The neurons are independent, each would determine if cat / not, dog / not, etc. So this is a good architecture for multi-hot since each neuron can focus on one of the categories.

### Encoding

* The first neuron layer will be sensitive to simple patterns, the next layer will be sensitive to more complex features like eyes or ears. The second layers receives information that is more complex than the pixels received by the first layer. Third layer, etc represents even more complex features
* This is called encoding - each layer encodes different levels of features.

### Day / Night Classification 

* Given image, label day (0) or night (1)
* Data - could have labeled pictures of day and night, balance data set. 10k images a good start, but wouldn’t wait to gather this many to start working on the problem. Could go to pixel bay and just search for day and night images, might have mistakes but with a big enough data set would be good enough
  * Indoor pictures in general are going to be difficult. Also edge cases at dawn dusk, etc and 10k won’t be enough to resolve the edge cases
* Input - do we care about real time, then we have to consider speed. Even if not training time might matter. Low resolution images are probably fine for this problem since we aren’t looking at granularity and will make training faster. 
  * What is low resolution? Consider orders of magnitude. For face recognition 400 x 400 is pretty classic. In this scenario 64 x 64 is enough, but RGB important here. If you aren’t sure, just see if a human can distinguish with different resolutions with accuracy - this is a good guide to what a model might need.
* Output - 0 or 1. If we wanted more specificity for time of day could use one-hot, softmax instead of sigmoid. 
* Architecture - shallow network probably works well
* Loss function - binary cross entropy, since this is standard for binary classification tasks.

### Face Verification

* School wants to use face verification for validating student IDs when students swipe their cards. Do they match their ID pictures?
* Data - University has labeled images of every person with a card. But this is not enough, need more generalizability in the model. 
* Input - give camera picture and stored picture 0 or 1 if same person or not. Resolution ~ 400 x 400
* Output - 0 it is the same person, 1 different
* Architecture - say the simplest idea could be compute distance pixel by pixel. Issues with brightness, background colors, faces aren’t centered the same way, face could change with aging hair etc.
  * Solution - use encoding. Run the id image and the camera image through the network. The vectors in the network should have much more information than the pixels on their own. Distance of the vectors should be more meaningful than pixel distance
* How do we train this kind of network? Triplets (Anchor, Positive, Negative)
  * What we want is pictures of the same person to have similar encoding and pics of different people should have different encodings. So we generate triplets - an anchor picture, the original picture. A positive, the same person. A negative, a different person. Now we want to min distance for anchor - positive and max distance anchor - negative. 
  * Minimize loss function $L = ||Enc(A) - Enc(P)||^2_2 - ||Enc(A) - Enc(N)||^2_2$
* Now we modify the problem - no swipes, the camera just identifies you from the camera. Could use KNN from the DB of faces and compare vectors, could require more than one match to make more robust.
* Another tweak - we want face clustering, all photos of a single person together. Apply K-means clutering to vectors and see given a new picture which group does it fall into, each group is a person.

### Neural Style Transfer

* Taking a picture, make it beautiful
* Data - any data
* Input is a content image and style image - want the ouput to be a generated image that is the content image in the style of the style image
* Architecture - content is probably something you can get from lower level encoding of a good network. We use a pre-trained model because it extracts important information from images. Extract a content vector C. Feed in the style image, and find a deeper encoding to find a Gram matrix - style S. For image generation - feed in random image, pull out content CG and style SG, compute loss. After many iterations should get image with altered style. Loss: $L = ||Style_S - Style_G||_2^2 + ||Content_C - Content_G||_2^2$
* The parameters are not being tuned - the network is fixed. It’s the pixels that are tweaked to minimize the loss functions - $\frac{dL}{dx}$

### Trigger Word Detection

* Given 10 sec audio speech, detect word activate.
* Data - variety of settings/people saying activate, and audio of saying other things. Need a wide distribution of accents, silence, noise
* Input - audio clips where the segment that is activate is specified. Can consider resolution - look at papers / experts for sample rate
* Output is 0, 1? Want 0 or 1 at the spot of the word, not for the whole audio clip. This makes for much more efficient training. Another problem is huge imbalance between positives and negatives. Sequential sigmoid last activation
* Architecture - RNN
* Critical piece is data collection and labelling process. Instead of manual collection and labeling, create DB of positive words, negative words, background noise. BN is freely available online, no problem. Record in the world, just get word by word recordings - just positive word and just negative words. No need to label then, can just write script to insert positive and negative words into background noise, script can automatically label the data. By recombining in different ways, get millions of data points. The way you collect data is critical.

## Full Cycle Deep Learning

### Steps in a Project

* 1) Select a project. Example: Using Facial Recognition to Unlock Doors
* Get data, design model, train and evaluate the model and iterate, ship and deploy. Finally maintain the system

### Getting Data

* How many days will you use to collect data? Important to get some data, but don’t wait to long to dive into the problem. Only need a small amount to get started, see how far we get, then collect more data dependent on performance. 1-2 could easily suffice for an initial dataset. 
* Get data quickly and train a quick model. Doesn’t need to be the most up to date, complicated network. Find something OSS on Github, etc. Look at the results to determine how far you are from the objective. Iterate over this process, slowly improving the dataset and the model.
* Note - this is reasonable for jumping into a new domain. If you already have the domain knowledge, you might already know that you need a minimum number of examples, or a certain model architecture. 

### Deploying Model

* Edge device - model is running on the physical device. Cloud device - streams the data to the cloud for processing. 
* For door problem, cannot have 30 f/s through NN or 24 hour streaming video to the cloud. Instead might feed an image to an activity detector as preprocessing. Most of the time, nothing is changing outside of the front door, no need to run the inference engine during this time. 
* Only when some activity is detected do you feed it to the NN for classification.
* How should we deploy an activity detector?
  * Could write a program to take in a picture from 10s ago and current picture, sum up differences in pixels, and trigger for some threshold.
  * Alternatively, train a small NN to predict human or not.
  * Option 1 could lead to higher errors, option 2 could pass fewer images to the higher compute NN. Option 1 could be written quite quickly given its simplicity, option 2 has more to tune. 
  * Option 1 can be a good choice for the quick and dirty model. Option 2 might be a backup if we see that option 1 is a problem. Also has very few parameters to tune. Consistent with the theme to do something quick and dirty first to see if it works, then iterate upwards when it does not.
  * The other thing to consider is that the data changes in ML systems. Weather changes with seasons, people dress differently, people look different in other regions, hardware used might change, diversity of population, etc. A network trained to recognize data at a point in time and location will not be robust

### Model Maintainence

* Web search - training a system of search ranking, say there is a new figure, celebrity or language changes. The ranking algorithm no longer is relevant because the world has changed.
* Speech recognition - trained on adult voices, but younger people were more likely to use speech recognition and the younger voices had different performance. Speech recognition attempted to be used in noisier environments, cars, initial dataset did not cover these situations.
* Defect inspection - say in manufacturing, detecting scratches on smart phones. If the lighting changes in the factory, algorithm may cease working.
* When the data changes, have to get new data, retrain the model, redeploy. Using a simple threshold / preprocessing step, it is easy to update or retune this section. Using a neural network means more retraining before you even get to the heart of the model.
* You also want to get ahead on maintainence - deploy updates before users complain. Between cloud and edge approaches to deployment, cloud is easier to maintain. When we deploy an ML system, often set up monitors / dashboard where we can monitor key metrics over time with threshold bands, notified once we exceed those thresholds. 
* Often can just sample some portion of the data from devices as a statistical sample to monitor things. Cloud vs edge doesn’t affect the day 1 accuracy of the system, but if the system is designed to continue to accumulate data and improve will help keep the system working.  Builds a defensive moat to ward off competitors.
* QA: In ML, our problems are statistical more than software bugs. The testing is not binary, right or wrong, but some % accuracy on the test set. Always important to ensure you meet some accuracy criteria.

# Coursera Modules

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


### Module 3 - Shallow Neural Networks

* Starting with a single hidden layer NN. Let $a = x^{[0]}={x_1, x_2, x_3}$ be the input layer, fully connected to the hidden layer with 4 nodes. Those in turn connect to the output layer, with a single node, whose output is $\hat{y} = a^{[2]}$. Superscript bracket notation - the layer of the NN
* For four unit hidden layer: $a^{[1]} = \left[\begin{array}{c}a^{[1]}_1\\a^{[1]}_2\\a^{[1]}_3\\a^{[1]}_4\end{array}\right]$. 2 layer NN, because we do not count the input layer. Each $a^{[1]}_i$ has an associated $w^{[1]},b^{[1]}$
* Idea should be like the logistic regression, but simply repeated many times. First perform $z = w^Tx + b$, then $a=\sigma(z)$. In the NN with layers, have $z^{[1]}_i = w^{[1]T}_ix + b_i^{[1]}$  and $a^{[1]}_i = \sigma(z_i^{[1]})$ for layer 1, node i. Repeat for each node in the layer - but clearly we are going to vectorize these equations.
* Vectorizing, each $w^T_i$ is a row in the W matrix, so for layer 1 stack the equations for the different nodes: $W^{[1]T}\left[\begin{array}{c}x_1\\x_2\\x_3\end{array}\right] + \left[\begin{array}{c}b^{[1]}_1\\b^{[1]}_2\\b^{[1]}_3\\b^{[1]}_4\end{array}\right]=\left[\begin{array}{c}w^{[1]T}_1x + b_1^{[1]}\\w^{[1]T}_2x + b_2^{[1]}\\w^{[1]T}_3x + b_3^{[1]}\\w^{[1]T}_4x + b_4^{[1]}\end{array}\right] =\left[\begin{array}{c}z^{[1]}_1\\z^{[1]}_2\\z^{[1]}_3\\z^{[1]}_4\end{array}\right]=z^{[1]} $. Similarly $a^{[1]}=\left[\begin{array}{c}a^{[1]}_1\\a^{[1]}_2\\a^{[1]}_3\\a^{[1]}_4\end{array}\right]=\sigma(z^{[1]})$
* Taking the next layer, can perform the same operations, though the dimensions of the matrices can change depending on the input / output size.

##### Vectorized Implementation across Training Data

* Each $x^{(j)} \rightarrow a^{[2](i)}=\hat{y}^{(i)}$ - eg example i and layer 2 output. We run these layer equations over each i training examples $
  z^{[1](i)}=W^{[1]} x^{(i)}+b^{[1]},\;
  a^{[1](i)}=\sigma\left(z^{[1](i)}\right),\;
  z^{[2](i)}=W^{[2]} a^{[1](i)}+b^{[2]},\;
  a^{[2](i)}=\sigma\left(z^{[2](i)}\right)
  $
* Each training example is a column in the matrix X - n x m matrix. To vectorize over all examples and noting $X = A^{[0]}$, $Z^{[1]}=W^{[1]} A^{[0]}+b^{[1]},\;
  A^{[1]}=\sigma\left(Z^{[1]}\right),\;
  Z^{[2]}=W^{[2]} A^{[1]}+b^{[2]},\;
  A^{[2]}=\sigma\left(Z^{[2]}\right)$ - same equations just over matrices. 
* $Z^{[i]}$ is a matrix with each row is a node in the layer with examples running from 1 to m. $A^{[i]}$ is a matrix with each row a node in the layer with columns ranging 1 - m. For Z and A, columns are different training examples and vertically we have different nodes. For X, columns are different training examples and vertically we have different features. 
* $W^{[1]}x^{(1)}$ gives you a column vector, true for each training example. Instead we place each column x in X to get the vectorized version.

##### Activation Functions

* Activation functions can be different for layers, so may write $g^{[1]}, g^{[2]}$ to specify the layer associated with each activation.
* Sigmoid $a=\frac{1}{1+e^{-z}}$ on [0, 1]. Never use on hidden layers besides output layer. Others are strictly superior otherwise. 
  * Derivative $\frac{d}{d z} g(z) = \frac{1}{1+e^{-z}}\left(1-\frac{1}{1+e^{-z}}\right) = g(z)(1-g(z))$. In a NN, this would be equivalent to $a(1-a)$. Can see that if z = 10, $g(z) \approx 1,\;g’(z)\approx 0$
* Tanh $a=tanh(z)=\frac{e^{z}-e^{-z}}{e^{z}+e^{-z}}$ on [-1,1]. Similar shape as sigmoid with different range.
  * Almost strictly superior to sigmoid bc the range allows for a mean 0 - centers the data so learning for the next layer is a bit easier.  One exception is the output layer, where you might want the output to be a 0 or 1 for classification. However when z is very large or small, then the slope becomes very small as well, slowing learning in gradient descent.
  * Derivative: $\frac{d}{d z} g(z)=1-(\tanh (z))^{2}$. In NN a = g(z) and $g^{\prime}(z)=1-a^{2}$
* ReLU - $a=max(0,z)$ on $[0,\infty]$. 
  * Discontinuity at 0, but essentially not a problem, can just hard code or pretend derivative is 1 or 0. When z is negative, derivative is 0, but could modify to maintain a slight positive slope for z negative - Leaky ReLU. Advantage to either is that for the majority of z’s, the slope is very different from 0 (z > 0), allowing faster learning.
  * Derivative: $g^{\prime}(z)=\left\{\begin{array}{ll}
    {0} & {\text { if } z<0} \\
    {1} & {\text { if } z>0} 
    \end{array}\right.$ . Can in practice ignore 0, since change z is exactly 0 is near 0.
* Leady ReLU - $a = max(0.01z, z)$ on $[-\infty,\infty]$. Can modify the z coefficient or make it part of the learning function.
  * Derivative: $g^{\prime}(z)=\left\{\begin{array}{ccc}
    {0.01} & {\text { if }} & {z<0} \\
    {1} & {\text { if }} & {z>0}
    \end{array}\right.$
* If you aren’t sure which will work best, no problem in trying different ones and testing against a validation set.
* Why do we need a non-linear activation function? Without one, $\hat{y}$ is simply a linear function of X, can just plug in from dependent equations and refactor to see this. Could potentially use a linear function at the output layer, but otherwise definitely want something non-linear.

##### Gradient Descent

* Parameters $W^{[1]}, b^{[1]}, W^{[2]}, b^{[2]}$
* Cost function $J(W^{[1]}, b^{[1]}, W^{[2]}, b^{[2]}) = \frac{1}{m}\sum_{i=1}^n L(\hat{y},y)$
* Repeat: Compute predictions $\begin{array}{l}
  \hat{y}^{(i)}
  \end{array}, i=(\ldots, m)$, calculate derivatives of J wrt W and b for each layer, update each parameter.
* Forward propogation - the formulas derived above for Z, A for each layer.
* Back propogation:
  * Summary: single example equations:
    * $\begin{aligned}
      &d z^{[2]}=a^{[2]}-y\\
      &d W^{[2]}=d z^{[2]} a^{[1]^{T}}\\
      &d b^{[2]}=d z^{[2]}\\
      &d z^{[1]}=W^{[2] T} d z^{[2]} * g^{[1]{\prime}}\left(z^{[1]}\right)\\
      &d W^{[1]}=d z^{[1]} x^{T}\\
      &d b^{[1]}=d z^{[1]}
      \end{aligned}$
  * Summary: vectorized equations:
    * $d Z^{[2]}=A^{[2]}-Y$
    * $d W^{[2]}=\frac{1}{m} d Z^{[2]} A^{[1]^{T}}$ - 1/m comes from cost function J, now that we are averaging the losses
    * $d b^{[2]}=\frac{1}{m} n p . s u m\left(d Z^{[2]}, \text { axis }=1, \text { keepdims }=\text { True }\right)$
    * $d Z^{[1]}=W^{[2] T} d Z^{[2]} * g^{[1]{\prime}}\left(Z^{[1]}\right)$ - Note $\times$ is an element-wise product. 
    * $\begin{array}{l}
      {d W^{[1]}=\frac{1}{m} d Z^{[1]} X^{T}} \\
      {d b^{[1]}=\frac{1}{m} n p . \operatorname{sum}\left(d Z^{[1]}, \text { axis }=1, \text { keepdims }=\text { True }\right)}
      \end{array}$

##### Random Initialization

* Initializing the weights of the parameters - cannot simply initialize to 0
* Say a network with n=2, 2 hidden nodes and 1 output layer
* We can initialize the bias to 0’s without a problem
* If $W^{[1]} = \left[\begin{array}{ll}
  {0} & {0} \\
  {0} & {0}
  \end{array}\right]$ then $a^{[1]}_1 = a^{[1]}_2$, $dz_1^{[1]} = dz_2^{[1]}$ - our hidden units are identical and compute the same function. The whole network is symmetric, and will update the weights exactly the same at each iteration. W remains a rank-1 matrix through every iteration.
* Instead $W^{[1]} = np.random.randn((2,2))\times 0.01$, $b^{[1]} = np.zero((2,1))$, same for $W^{[2]}, b^{[2]}$. We want to initialize the weights to small values, since large values of W make large values of z and will be far out on the activation function ->  the derivative will be very small and learning will be slow. 
* We can choose a different constant than 0.01 for different situations - will be important in deeper learning.

### Module 4 - Deep Neural Networks

##### Deep L-layer NN

* Define variable L = # of layers, so last layer = $n^{[L]}$, $\hat{y} = a^{[L]}$
* $n^{[l]} = $ # units if layer l. $a^{[l]}$ = activations in layer l. $a^{[l]} = g^{[l]}(z^{[l]})$,
* Forward propogation
  * $z^{[1]} = W^{[1]}x + b^{[1]},\; a^{[1]} = g^{[1]}(z^{[1]})$
  * Generally $z^{[l]} = W^{[l]}a^{[l-1]} + b^{[l]},\; a^{[l]}=g^{[l]}(z^{[l]})$
  * etc down to layer 4 for L = 4, and the final variable $a^{[4]} = \hat{y}$
  * For vectorized formulae: $Z^{[1]} = W^{[1]}A^{[0]} + b^{[1]},\; A^{[1]}=g^{[1]}(Z^{[1]})$ then $\hat{Y} = A^{[4]}$
  * Note we actually do loop over the layers to iteratively calculate the activations for each layer. Here `for l in range(1,4)`

##### Matrix Dimensions

* One Vector
  * For example L = 5
  *  $n^{[0]}=2,\;n^{[1]}=3,\;n^{[2]}=5,\;n^{[3]}=4,\;n^{[4]}=2,\;n^{[5]}=1$
  * $z^{[1]} = (n^{[1]}, 1)\leftarrow X = (n^{[0]}, 1)$ so W must be $W^{[1]}: (n^{[1]}, n^{[0]})$
  * W is a transformation matrix from X space to Z space. 
  * General formula, $W^{[\ell]}:\left(n^{[\ell]}, n^{[\ell-1]}\right)$ and $b^{[\ell]}: (n^{[\ell]}, 1)$
  * For backprop: $dW^{[l]}: (n^{[l]}, n^{[l-1]})$, $db^{[l]}: (n^{[l]}, 1)$
* Vectorized Matrix Dimensions
  * $Z^{[l]} = (n^{[1]}, m)$, $X = (n^{[0]}, m)$, $W^{[1]}: (n^{[1]}, n^{[0]})$
  * $b^{[\ell]}: (n^{[\ell]}, m)$ by broadcasting
  * $Z^{[l]},\; A^{[l]}: (n^{[l]},m)$

##### Deep Representation

* Deeper layers represent more complex expressions - encoding
* You can compute some functions with small L-layer deep neural network that shallower networks would require exponentially more hidden units to compute. Imagine making an XOR tree computing XOR between a string of variables, can perform it pairwise in a tree similar to a NN - depth of network $O(log(n))$. With just one hidden layer, this network would need to consider $2^n$ combinations in one layer.

##### Building Blocks of NN

* When going forward input $a^{[l-1]}$ has output $a^{[l]}$. Cache $z^{[l]}, W^{[l]}, b^{[l]}, a^{[l-1]}$
* For backprop, input $da^{[l]}$ has output $da^{[l-1]}, dW^{[l]}, db^{[l]}$, cache($z^{[l]}$)
* The cache contains the Z functions, since the actual output is a, Z after activation. We need the cache to compute the derivatives.
* ![Screen Shot 2020-01-14 at 1.57.28 PM](/Users/spencerbraun/Documents/Notes/Stanford/CS230/Screen Shot 2020-01-14 at 1.57.28 PM.png)
* Backprop formulas for any layer:
  * $d Z^{[l]}=dA^{[l]} * g^{[1]{\prime}}\left(Z^{[1]}\right)$
  * $d W^{[1]}=\frac{1}{m} d Z^{[1]} A^{[l-1]T}$
  * $d b^{[l]}=\frac{1}{m} n p . \operatorname{sum}\left(d Z^{[l]}, \text { axis }=1, \text { keepdims }=\text { True }\right)$
  * $dA^{[l-1]} = W^{[l]T}dZ^{[l]}$
  * When doing logistic regression, input for backdrop $da^{[l]} = -y/a + \frac{(1-y)}{(1-a)}$

##### Parameters and Hyperparameters

* Hyperparameters - learning rate, # iterations, # hidden layers L, # hidden units $n^{[i]}$, activation function
* We need to tell our learning algorithm these features. They control the parameters learned on the data
* We will add to this list - momentum, mini batch size, regularizations, ...
* Need to iterate through trying values for the hyperparamters to see their effect on the model.

## C2 - Hyperparameter tuning, Regularization and Optimization

### Module 1 - Practical Aspects of Deep Learning

##### Training / Dev / Test Sets

* So many hyperparameters to determine - layers, hidden units, learning rates, activation functions. This is why deep learning is such an iterative process. Often intuitions in one application area do not transfer well to another application
* Break data into training, validation / dev set, finally a test set. Use the validation set for CV and tuning, finally get results from test set.
* In the big data era, dev and test sets become much smaller % of total data than classic 70/30 split. Say for 1mm examples, might just keep 10k for dev, 10k test.
* Mismatched training and dev / test sets. If you are pulling data from different sources, this will turn out poorly. Make sure dev and test are definitely from the same distribution, but many people will pull in additional training data that may come from different sources that may exhibit other features.
* Not having a test set might be ok - only needed if you want an unbiased estimate of your error. 

##### Bias - Variance

* Flexible - high variance, Rigid - high bias.
* Key numbers are the train set error and dev set error. Great train set error but poor dev set error - we have high variance and have likely overfit.
* If instead, train set error is 15% and dev set error is 16%, assuming humans have good performance on this task, the algorithm is underfitting the data and has high bias.
* If instead 15% on training error and 30% on dev set error - have high variance and high bias. Worst of both worlds. Can construct high bias and high variance models that are highly linear in some areas while highly overfit in other regions - happens most in high dimensional inputs.
* All of this predicated on knowing the human error or optimal Bayes error rate. If instead Bayes were higher, then we would have to adjust how we interpret these numbers

##### Recipe for ML

* Bias: After training, do we have high bias? Look at the training performance. If we are not fitting the training too well, could increase the network size, train longer, change our architecture. Repeat trying solutions until our bias is reduced and we fit the training data pretty well.
* Variance: Next look at the dev set performance. If we have high variance, can we get more data, regularize, try another NN architecture? 
* Notice these bias and variance solutions can be quite different - important to diagnose your problem correctly.
* Bias-Variance Tradeoff in DL - so long as you can make the NN deeper and get more data, we can reduce the bias or variance respectively without really any tradeoff from the other measure. 

##### Regularization

* Frobenius norm: $\left\|w^{[l]}\right\|^{2}=\sum_{i=1}^{n^{l}} \sum_{j=1}^{n^{[l-1]}}\left(w_{i, j}^{[l]}\right)^{2}$
* If you have high variance / overfitting, regularization is a good alternative to getting more data
* L2 norm for a vector: $\|\omega\|_{2}^{2}=\sum_{j=1}^{n_{x}} \omega_{j}^{2}=w^{\top} w$
* L2 regularization: $J(\omega, b)=\frac{1}{m} \sum_{i=1}^{m} L\left(\hat{y}^{(i)}, y{(i)}\right)+\frac{\lambda}{2 m}\|\omega\|_{2}^{2}$ - note we omit b from regularization
* L1 $\frac{\lambda}{2 m}\|w\|_{1}$ instead will produce a sparse w, but in practice L2 is used much more often.
* In a neural network, we are dealing with matrices - $\frac{1}{m} \sum_{i=1}^{n} L\left(\hat{y}^{(i)}, y^{(i)}\right)+\frac{\lambda}{2 m} \sum_{l=1}^{L}\left\|\omega^{(l)}\right\|^{2}_F$
* In gradient descent, we calculated dW from backprop, then update W
* Now we calculate the same dW, but add $\frac{\lambda}{m} \omega^{[l]}$. This is also called weight decay, since it is equivalent to multiplying W by a factor less than 1. 
* Imagine fitting a large and deep NN, and it is overfitting the data. Our regularizer penalizes large weight matrices, so many of the hidden units carry small weights and the network is equivalent to a simpler NN with fewer hidden units.
* Additionally, look at the tanh activation function. With a large regularization parameter, z will be constrained to be small as well. This puts in the in linear range of tanh around 0 with a strong slope - then every layer is roughly linear, and our model will have a near linear decision boundary.
* In gradient descent, make sure you are plotting the new definition of the cost function J - otherwise may not see a monotonic decrease as we expect.

##### Dropout Regularization

* Randomly eliminate a set of nodes across different layers. We are now training a smaller network on each example.
* Inverted Dropout - a method of implementing dropout, say with l=3. d3 is the dropout vector for layer 3: `d3 = np.random.rand(a3.shape[0], a3.shape[1]) < keep_prob` for `keep_prob= 0.8`, the probability that a unit will be kept in the model. Then there is a 20% chance that the node in the layer will be dropped. Then `a3 = np.multiply(a3,d3)`, then d3 will zero out the entries of a3 that correspond by the dropout. Finally `a3 /= keep_prob` to reinflate the a3 weights to ensure the expected value of a3 remains the same.
* At test time, we do not use dropout and run the test points on the full network. This would just add noise to your predictions, since the dropout is part of random process.
* Note: In general, the number of neurons in the previous layer gives us the number of columns of the weight matrix, and the number of neurons in the current layer gives us the number of rows in the weight matrix.
* Intuition: we cannot put too much weight on any given unit, since it may not be present in the next iteration of training. Weights become spread over more units, reducing the size of weights on any given unit.
* `keep_prob` can vary by layer. The largest hidden layer would have the largest weights, so you might set keep_prob to be lower than other layers where we don’t worry about overfitting. Smaller layers, we probably want minimal if any dropout.
* Computer vision makes heavy use of dropout since they almost never have enough data for the size of the network, but in other application areas would wait to see overfitting before trying dropout.
* Cost function J is less well defined and we lose the debugging tool of seeing a monotonically decreasing J in gradient descent. Can run the network without dropout just as a debugging tool to double check.

##### Other Regularization Methods

* Data augmentation - say with image, you could flip or mirror image your training data to increase data size. Not as good as new data but a good proxy. Could also apply random distortions or cropping. 
* Early stopping - as you run gradient descent, you plot the cost function J and it should decrease monotonically. With early stopping also plot the dev set error - this will be more of a U convex shape. Can stop training the neural network at the dev set min instead of minimizing J all the way. By stopping halfway, our W is smaller than the fully trained network.
  * Orthogonalization - you want to think about one objective at a time. Downside of early stopping is we are not considering optimizing J and preventing overfitting separately. Easier to search of all hyperparameters with L2 reg, but it is more computationally expensive since we have to search over the lambda space.

##### Normalizing Inputs

* Formula for normalization: $\frac{x-\mu}{\sigma}$ - 0 mean and standard variance. The variance along each predictor variable is 1
* We normalize same mean and variance to standardize test and training set
* If we didn’t normalize, cost function likely to be distorted to odd curve, which distorts the weighting values - elongated eliptical curves instead of circles. Gradient descent is likely to go much more directly to the minimum, instead of bouncing around.

##### Vanishing / Exploding Gradients

* Training difficult with very large and very small derivatives. Especially problematic in very deep networks - $y = W^{[l]}W^{[l-1]}...W^{[0]}x$. Note $a^{[1]} = W^{[0]}x$, $a^{[2]}=W^{[1]}W^{[0]}x$, etc. We are taking large powers of the weight matrices, so the value of y can explode for weights over 1 (W > I). For weights less than 1 ( W < I ), the exponents shrink the weights to zero.
* Learning slows for tiny gradients, takes too large steps for large.
* Weight Initialization Fix: Say we have 4 inputs in X. $z = w_1x_1 + ...w_nx_n$ - for larger n we want smaller w’s to balance out. Can set $Var(w_i) = 1/n$. In practice, set $W^{[l]} = $ `np.random.randn(slope)*np.sqrt(2/n^{[l-1]})` (better to set variance to 2 instead of 1). This is the fix generally for ReLU
* For tanh, we tend to use Xavier initialization: $\sqrt{\frac{1}{n^{[l-1]}}}$ or $\sqrt{\frac{2}{n^{[l-1]}+n^{[l]}}}$. These initializations for variances can be tuned as well.

##### Gradient Approximation and Checking

* Numerical approximation - better approximations of derivatives by taking triangle around point we are interested in: $\frac{f(\theta + \epsilon)- f(\theta - \epsilon)}{2\epsilon} \approx g(\theta)$ - notice this is the difference quotient in the limit. The error is on the order of $O(\epsilon^2)$ instead of one side epsilon with error on order of $O(\epsilon)$
* Gradient checking for debugging and verification. Take all parameters W, b, concatenate and reshape into a big vector $\theta$. Take the dW, db, etc into a big vector $d\theta$. 
* $J(\theta) = J(\theta_1, \theta_2,...)$. For each i, $d\theta_{approx}[i] = \frac{J(\theta_1,...,\theta_i + \epsilon,...) - J(\theta_1,...,\theta_i - \epsilon,...)}{2\epsilon} \approx d\theta[i] = \frac{\partial J}{\partial \theta_i}$
* We take $\frac{||d\theta_{approx} - d\theta||_2}{||d\theta||_2 + ||d\theta||_2} \approx 10^{-7}$ is a good range for the approximation to be close to the expected gradient. If it climbs to $10^{-3}$, should be cause for worry.
* This is not used for training, just debugging - backprop is reserved for actual training. 
* If algorithm fails grad check, look at components, see which have the largest differences. Remember if you included regularization, you need to include it in the grad check (note this won’t work for dropout since it is randomized). Run at random initialization, then try again after some training since the problem may be cause by extremely small weights that will grow over time.

### Module 2 - Optimization Algorithms

##### Mini-batch Gradient Descent

* Vectorization allows you to compute on m examples. If m = 5mm, we have to process the whole dataset to take a single step in gradient descent
* If we split up the training set into mini-batches of say 1000, and call these batches $X^{\{1\}}, X^{\{2\}},...$ for $X^{\{1\}} = ^{(1)},...,x^{(1000)}$. Then we have mini-batch t $X^{\{t\}},Y^{\{t\}}$
* Running mini-batch gradient descent - for t = 1,...,5000 we take 1 step of grad descent using $X^{\{t\}},Y^{\{t\}}$, vectorizing calcs over those 1000 examples in each mini-batch. Perform backprop to compute gradients for the batch - this is 1 epoch of training.
* Here, the cost function may not monotonically decrease, since we minimize over new training batches at each epoch. Should be jittery, but trend downwards. Some mini-batches may be easier to minimize than others, leading to non-monotonic minimization.
* Choosing mini-batch size - if batch size = m, we have batch gradient descent. At the other extreme, could set batch size to 1, producing stochastic gradient descent where every example is its own mini-batch $X^{\{t\}},Y^{\{t\}} = x^{(t)},y^{(t)}$
  * SGD won’t march directly to the minimum like BGD. Examples may lead it in an oscillating or jittery path. 
  * In practice, use something in between. BGD takes too long to process the large dataset at each step. SGD loses all the speedup from vectorization (note the jitter can be controlled via learning rate). MBGD gives us the fastest learning in practice; we get vectorized speed ups while making progress with smaller datasets.
  * If m less than 2000, just use BGD. Otherwise often use MB sizes of 64, 128, 256, 512. Make sure MBs fit in CPU / GPU memory

##### Exponentially Weighted Averages

* Say the temperature in a city over time - cold in the winter, warm in the summer. Might be useful to have a moving average. Initialize $V_0 = 0,\; V_1 = 0.9 V_0 + 0.1\theta_1$ and generally $V_t = \beta V_{t-1} + (1-\beta)\theta_t$
* $V_t$ is approximately average over $\approx \frac{1}{1-\beta}$ days’ temperature. With a high value of beta, our curve gets much smoother, but the curve shifts as it take more days of changes to move the curve - it reacts much more slowly to changes day over day. Similarly, over smaller beta like 1/2, we average over a small number of days and the curve is highly variable
* Notice this formulas are recursive. We essentially have an exponentially decaying function in the thetas of prior days. Note $(1-\varepsilon)^{1/\varepsilon}=\frac{1}{e}$ - use this as the gauge to see how far back weights hold power for a given epsilon/beta.
* Implementation - initialize, V=0, then each day update V $V_{\theta} := \beta V_{\theta} + (1-\beta)\theta_t$. This takes very little memory to compute this weighted average. 
* Bias correction - when we initialize to 0, we get a poor estimate of the first data points. For day t, use $\frac{V_t}{1-\beta^t}$ instead. When t is large, denominator becomes 1, but for small t, the denominator boosts the initial values.

##### Gradient Descent with Momentum

* Compute exponentially weight average of your gradient and use this to update your weights.
* If you have elliptical level curves, GD tends to oscillate back and forth, forcing us to use a slower learning rate. On one axis (shorter) we want slower learning and the other (longer) axis want faster learning.
* Momentum  - on iteration t, compute dW, db on current MB. Compute $V_{dW} = \beta V_{dW} + (1-\beta)dW$, the MA for the derivatives of W. $V_{db} = \beta V_{db} +(1-\beta)db$. Then $W = W - \alpha V_{dW}$, $b := b- \alpha V_{db}$. In long axis direction, derivatives are aligned and GD moves quickly in this direction, but in the shorter axis derivates cancel out and the moving average dampens the oscillations. 
* The $dW$ term is like the acceleration and $V_{db}$ is the velocity. The beta acts as friction on the ball rolling down the bowl.
* $\beta$ is a new hyperparameter we tune alongside $\alpha$, though commonly beta is 0.9, the average over the last 10 gradients. In practice, most do not perform bias correction, since within ten iterations we should have a good approximation for the gradient. Additionally, often omit $(1-\beta)$, and run $V_{dW} = \beta V_{dW} + dW$, though this is a bit less intuitive.

##### RMSprop

* Root mean squared prop. Speed up learning on the long axis, slow it on the short. 
* On iteration t, compute dW, db on current MB. $S_{dW} = \beta_2 S_{dW} + (1-\beta_2)dW^2$ where square is element wise operation. $S_{db} = \beta_2 S_{db} + (1-\beta_2)db^2$. Then $W = W -\alpha \frac{dW}{\sqrt{S_{dW}} + \epsilon}$, $b = b - \alpha\frac{db}{\sqrt{S_{db}+\epsilon}}$. If we say W is along the long direction, it will have small dW, and b along short direction will have big db. Net effect is the the short axis direction, we divide by a larger number dampening updates and the opposite for the long direction. In practice we are in higher dimensions, could be some W vector. 
* Note the epsilons are to ensure numerical stability - ie. not dividing by zero or a number close to zero.

##### Adam Optimization

* Adaptive Moment Estimation. Init: $V_{dW} = 0$, $S_{dW} =0 ,\; V_{db}=0,\;S_{db} = 0$
* On iteration t: compute dW, db using current MB. Use momentum and RMSprop equations to update the parameters above. 
* Implement bias correction $V_{dW}^{corrected} = V_{dW} / (1-\beta^t_1),\; V_{db}^{corrected} = V_{db} / (1-\beta^t_1),\; S_{dW}^{corrected} = S_{dW} / (1-\beta^t_2),\; S_{db}^{corrected} = S_{db} / (1-\beta^t_2)$
* Then to update parameters - $W := W - \alpha \frac{V_{dW}^{corrected}}{\sqrt{S_{dW}^{corrected}}+\epsilon}$ and $b := b - \alpha \frac{V_{db}^{corrected}}{\sqrt{S_{db}^{corrected}}+\epsilon}$
* Many hyperparameters: $\alpha$ needs to be tuned, $\beta_1: 0.9,\;\beta_2:0.999,\;\epsilon:10^{-8}$ 

##### Learning Rate Decay

* Slowly reduce your learning rate over time. Formula: $\alpha=\frac{1}{1+d e c a y R a t e \times \text { epoch } N u m b e r} \alpha_{0}$
* Algorithm in GD may not fully converge to minimum, keeps taking steps around the min. If we reduce alpha over time, we oscillate in a tighter region around the min as we get closer. We can afford to take larger steps at the beginning and want smaller steps when we are close.
* Can try a variety of values for hyperparameters $\alpha_0$ and decay rate
* Could also rate decay in other ways. For example $\alpha= 0.95^{\text{epoch num}}\alpha_0$  or $\alpha = \frac{k}{\sqrt{\text{epoch num}}}\alpha_0$, discrete stepwise function, manual decay
* Learning rate decay likely lower down on the list of things to try to improve your NN

##### Local Optima and Saddle Points

* In high dimensions - actually most points of zero gradients are not local optima but are saddle points. This is because you would need all p dimensions forming a cup or cap together, but much more likely to have some dimensions with positive and others with negative derivatives. 
* Problem of plateaus - areas of zero for a large portion of a surface. This means it will take a long time to find a way towards an area of more extreme gradient. 

