# Machine-Learning
Machine Learning Module

# `Data Mining by Fayyaz Minhas

##


## This document includes:

- The link to the module video playlist
- Books
- Detailed Weekly breakdown of module contents
- Self Assessment Revision Questions

## Video Playlist: [https://www.youtube.com/playlist?list=PL9IcorxiyRbASB9DXjoWnBJO9RSKyzM2N](https://www.youtube.com/playlist?list=PL9IcorxiyRbASB9DXjoWnBJO9RSKyzM2N)

## Books

[PML] Probabilistic Machine Learning: An Introduction by Kevin Patrick Murphy. MIT Press, 2021. link: [http://mlbayes.ai/](http://mlbayes.ai/)

[IML] Introduction to Machine Learning 3e by Ethem Alpaydin (selected chapters: ch. 1,2,6,7,9,10,11,12,13)

[DBB] Deep Learning by Ian Goodfellow, Yoshua Bengio, Aaron Courville, (Ch 1-5 if needed as basics), Ch. 6,7,8,9 link: [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)

[FNN] Fundamentals of Neural Networks : Architectures, Algorithms And Applications by Laurene Fausett, (ch. 2,6)

## Module Contents

| 1 | 1.1 Introduction1.2 Why Data Science? 1.3 Applications 1.4 Research Applications1.5 Framework | [Introduction Slides](https://warwick.ac.uk/fac/sci/dcs/teaching/material/cs909/minhas-ml-1-introduction.pdf)[Applications and Framework Slides](https://warwick.ac.uk/fac/sci/dcs/teaching/material/cs909/minhas-ml-2-machine_learning_framework.pdf)[k-Nearest Neighbor Algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) [Required][PML] Chapter-1, [IML] Chapter-1[CRISPR Talk](https://youtu.be/4TB4Z8_T3d8)[Whole Slide Images are Graphs Talk](https://youtu.be/Of1u0i7roS0)[Project Suggestions](https://warwick.ac.uk/fac/sci/dcs/teaching/material/cs907/suggestions/minhas_fayyaz)The master algorithm (casual reading)[A few useful things to know about machine learning](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf) | [Learning Python](https://warwick.ac.uk/fac/sci/dcs/teaching/material/cs909/#learningPython) |
| --- | --- | --- | --- |
| 2 | 2.1 Classification and Linear Discriminants2.2 Determining Linear Separability2.3 Prelim: Gradients and Gradient descent2.4 Prelim: Gradient Descent Code2.5 Prelim: Convexity2.6 Perceptron Modeling[Perceptron Code](https://web.microsoftstream.com/video/d84e1a2d-3e5e-4d05-b35f-ffac56c83025) | [Linear Discriminants (notes)](https://warwick.ac.uk/fac/sci/dcs/teaching/material/cs909/minhas-ml-3_linear_discriminants.pdf)[Preliminaries (notes)](https://warwick.ac.uk/fac/sci/dcs/teaching/material/cs909/gradients_gradient_descent_and_convexity.pdf)[Building Linear Models (notes)](https://warwick.ac.uk/fac/sci/dcs/teaching/material/cs909/minhas-ml-4_building_linear_models.pdf)[Gradient Descent Code (py)](https://github.com/foxtrotmike/CS909/blob/master/gd.py)[Perceptron Code (py)](https://github.com/foxtrotmike/CS909/blob/master/perceptron.py)[Perceptron Algorithm](https://en.wikipedia.org/wiki/Perceptron)
 | [Implementing kNN classifier](https://github.com/foxtrotmike/CS909/blob/master/DM_1_kNN.ipynb) |
| W | Lectures | Reading/Resources | Labs |
| 3 |
3.1 What&#39;s special in an SVM?3.2 SVM Formulation3.3 A brief history of SVMs3.4 Coding an SVM and C3.5 Margin and Regularization3.6 Linear Discriminants and Selenite Crystals3.7 Selenit Crystals bend Space3.8 Using transformations to fold3.9 Transformations change distance and dot products3.10 Kernelized SVMs |
[SVM Notes](https://warwick.ac.uk/fac/sci/dcs/teaching/material/cs909/minhas-ml-5_perceptron_to_svm.pdf)[SVM Applet](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)[Regularized Perceptron](https://github.com/foxtrotmike/CS909/blob/master/regper.ipynb)[Transformations code](https://github.com/foxtrotmike/CS909/blob/master/transformations.ipynb)[Fold and Cut Theoreom](https://youtu.be/ZREp1mAPKTM)Book Reading [SVM in PML, SVM in IML][SVM Tutorial](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000173) | [Gradient Descent and Perceptron](https://github.com/foxtrotmike/CS909/blob/master/dm_lab_2_fm.ipynb)(see lectures from previous week)[SVM](https://github.com/foxtrotmike/svmtutorial/blob/master/svmtutorial.ipynb)Assignment-1 Announced |
| 4 |
4.1 Scientific Method4.2 Why measure performance?4.3 Accuracy and its assumptions4.4 Confusion matrix and associated metrics4.5 ROC Curves4.6 PR Curves4.7 PR-ROC Relationship and coding4.8 Estimating Generalization4.9 CV in sklearn | Chapter 19 &quot;Design and Analysis of Machine Learning Experiments&quot; Alpaydin, Ethem. 2010. Introduction to Machine Learning. Cambridge, Mass.: MIT Press.[Lecture Notes](https://warwick.ac.uk/fac/sci/dcs/teaching/material/cs909/minhas-ml-6_evaluating_and_comparing_models.pdf)[Performance Assessment Exercise](https://github.com/foxtrotmike/CS909/blob/master/evaluation_example.ipynb)Optional ReadingWainberg, Michael, Babak Alipanahi, and Brendan J. Frey. &quot;Are Random Forests Truly the Best Classifiers?&quot; Journal of Machine Learning Research 17, no. 110 (2016): 1–5.Munir, Farzeen, Sadaf Gull, Amina Asif, and Fayyaz Ul Amir Afsar Minhas. &quot;MILAMP: Multiple Instance Prediction of Amyloid Proteins.&quot; IEEE/ACM Transactions on Computational Biology and Bioinformatics, August 22, 2019. https://doi.org/10.1109/TCBB.2019.2936846. | [SVM](https://github.com/foxtrotmike/svmtutorial/blob/master/svmtutorial.ipynb)[Performance Assessment Exercise](https://github.com/foxtrotmike/CS909/blob/master/evaluation_example.ipynb)Work on Assignment-1 |
| 5 | 5.1 Twelve ways to fool the masses5.2.1 Prelim: Old MacDonald meets Lagrange5.2. Prelim: Meet stubborn vectors5.2.3 Prelim: Covariance and its friends5.3 PCA | [Lecture Notes](https://warwick.ac.uk/fac/sci/dcs/teaching/material/cs909/minhas-da-5-pca.pdf)[Ten ways to fool the masses with machine learning](https://arxiv.org/abs/1901.01686)[Eigen Values and Vectors](https://github.com/foxtrotmike/PCA-Tutorial/blob/master/Eigen.ipynb)[PCA Tutorial](https://github.com/foxtrotmike/PCA-Tutorial/blob/master/pca-lagrange.ipynb) | [Eigen Values and Vectors](https://github.com/foxtrotmike/PCA-Tutorial/blob/master/Eigen.ipynb)[PCA Tutorial](https://github.com/foxtrotmike/PCA-Tutorial/blob/master/pca-lagrange.ipynb)Work on Assignment-1 |
| 6 |

6.1 Other dimensionality reduction methods6.2 SRM view of PCA6.3 OLSR6.4 OLSR to SVR6.5 Hurricane Intensity Regression | [Lecture Notes](https://warwick.ac.uk/fac/sci/dcs/teaching/material/cs909/minhas-ml-9_regression.pdf)
 | Work on Assignment-1(Assignment-2 posted [Download Assignment Description](https://warwick.ac.uk/fac/sci/dcs/teaching/material/cs909/cs909_2021_assignment_2.pdf)) |
| 7 |
7.1 How to go beyond classification, regression and dimensionality reduction7.2 Applied SRM in Barebones Pytorch7.3 Clustering7.4 Clustering in sklearn7.5 One class classifiers7.6 Ranking7.7 Recommender Systems7.8 Still more problems | [Lecture Notes](https://warwick.ac.uk/fac/sci/dcs/teaching/material/cs909/minhas-ml-10_other_machines.pdf)[Finding Anti-CRISPR proteins with ranking (optional)](https://www.youtube.com/watch?v=4TB4Z8_T3d8)[Using reinforcement learning to help a mouse escape a cat (optional)](https://youtu.be/N20h6vpR13Y) | [Barebones Linear Models](https://github.com/foxtrotmike/CS909/blob/master/barebones.ipynb)[Clustering](https://github.com/shaneahmed/StatswithPython/blob/main/clustering.ipynb)Work on Assignment-2 ([Download Assignment Description](https://warwick.ac.uk/fac/sci/dcs/teaching/material/cs909/cs909_2021_assignment_2.pdf)) |
| 8 | 8.1 Let me pick your brain8.2 Single Neuron ModelRevisit 7.2 Applied SRM in Barebones Pytorch8.3 Multilayer Perceptron8.4 Let&#39;s play with a neural network8.5 Deriving Backpropagation algorithm for MLPs8.6.1 MLP in Keras8.6.2 MLP in PyTorch using NN module8.6.3 MLP in PyTorch for MNIST with Dataloaders8.7 Improving learning of MLPs
 | [Lecture Notes](https://warwick.ac.uk/fac/sci/dcs/teaching/material/cs909/afsar-ml-13-week-8_deep_learning.pdf) | Work on Assignment-2 ([Download Assignment Description](https://warwick.ac.uk/fac/sci/dcs/teaching/material/cs909/cs909_2021_assignment_2.pdf))--[Barebones Linear Models](https://github.com/foxtrotmike/CS909/blob/master/barebones.ipynb)[Keras Barebones](https://github.com/foxtrotmike/CS909/blob/master/keras_barebones.ipynb)[NN module in Pytorch](https://github.com/foxtrotmike/CS909/blob/master/pytorch_nn_barebones.ipynb)[MNIST MLP in PyTorch](https://github.com/foxtrotmike/CS909/blob/master/pytorch_mlp_mnist.ipynb)[Trees and XGBoost](https://github.com/foxtrotmike/CS909/blob/master/trees.ipynb)Solve the XOR using a single hidden layer BPNN with sigmoid activations |
| 9 |

9.1 We can approximate the universe9.2 By going deep9.3 Finding Waldo is difficult with a fully connecteed MLP9.4 Convolution9.5 Learning filters9.6 CNNs9.7 CNN training in PyTorch9.8 Why CNNs9.9 CNN Hyperparameters and Regularization9.10 Transfer Learning and Application
 | [Lecture Notes](https://warwick.ac.uk/fac/sci/dcs/teaching/material/cs909/afsar-ml-13_week-9_deep_learning.pdf)[Deep PHURIE for hurricane intensity prediction](http://wrap.warwick.ac.uk/129159/) | Work on Assignment-2 ([Download Assignment Description](https://warwick.ac.uk/fac/sci/dcs/teaching/material/cs909/cs909_2021_assignment_2.pdf))-- Notes --[Universal Approximation Code](https://github.com/foxtrotmike/CS909/blob/master/uniapprox.ipynb)[Convolution in PyTorch](https://github.com/foxtrotmike/CS909/blob/master/pytorch_conv.py)[Learning a single convolution Filter](https://github.com/foxtrotmike/CS909/blob/master/learn_filters.py)[0 to AI in 10 lines of code](https://github.com/foxtrotmike/CS909/blob/master/0_to_AI_in_10_Lines_of_Python.ipynb) (By J. Pocock!)[Digit Classification with CNNs in Keras](https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/)[Digit Classification with CNNs in PyTorch](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/convolutional_neural_network/main.py)[Transfer Learning in PyTorch](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) |
| 10 | 10.1 Under the hood view of deep learning libraries10.2 Residual Networks (and other types)10.3 Autoencoders10.4 Generative Models10.5 GANs10.6 Barebones GAN in PyTorch10.7 Natural Language Modelling10.8 Using GANs for generating histology images (By S. Deshpande)10.9 Using Graph Neural Netwoks for histology images | [Lecture Notes](https://warwick.ac.uk/fac/sci/dcs/teaching/material/cs909/afsar-ml-13_deep_learning_complete.pdf) (All deep learning Notes)[GAN in PyTorch](https://github.com/foxtrotmike/CS909/blob/master/simpleGAN.ipynb)[NLP](https://www.analyticsvidhya.com/blog/2017/01/ultimate-guide-to-understand-implement-natural-language-processing-codes-in-python/)
 | Work on Assignment-2 ([Download Assignment Description](https://warwick.ac.uk/fac/sci/dcs/teaching/material/cs909/cs909_2021_assignment_2.pdf)) |

##


## Self Assessment Exercise Questions

### Week-2

Go through the lectures for Week-2 (uploaded on the module webpage) and then answer the following questions.

1. What is meant by a classifier? What is meant by the term &quot;generalization&quot;?

2. What is meant by a discriminant?

3. What is a linear discriminant? What is the mathematical representation of a linar discriminant?

4. What is meant by linear separability of a dataset?

5. Determine if each of the following datasets is linearly separable or not. If so, determine the linear discriminant for each dataset and calculate the score of the linear discriminant for each example in each dataset to verify your answer.

AND

(0,0) label: -1

(0,1) label: -1

(1,0) label: -1

(1,1) label: +1

OR

(0,0) label: -1

(0,1) label: +1

(1,0) label: +1

(1,1) label: +1

XOR

(0,0) label: -1

(0,1) label: +1

(1,0) label: +1

(1,1) label: -1

6. Determine if the following dataset is linearly separable or not. If so, determine the linear discriminant for this dataset and calculate the score of the linear discriminant for each of the following examples to verify your answer.

(0,0,0), label = +1

(0,0,1), label = -1

(0,1,0), label = -1

(0,1,1), label = -1

(1,0,0), label = -1

(1,0,1), label = -1

(1,1,0), label = -1

(1,1,1), label = -1

7. Determine if the following dataset is linearly separable or not. If so, determine the linear discriminant for this dataset and calculate the score of the linear discriminant for each of the following examples to verify your answer.

(0,0,0), label = +1

(0,0,1), label = -1

(0,1,0), label = -1

(0,1,1), label = -1

(1,0,0), label = -1

(1,0,1), label = -1

(1,1,0), label = -1

(1,1,1), label = +1

8. Determine if the following dataset is linearly separable or not. If so, determine the linear discriminant for this dataset and calculate the score of the linear discriminant for each of the following examples to verify your answer.

(0,0,0), label = +1

(0,0,1), label = -1

(0,1,0), label = -1

(0,1,1), label = +1

(1,0,0), label = -1

(1,0,1), label = +1

(1,1,0), label = +1

(1,1,1), label = -1

9. Do you think there can be multiple linear discriminant solutions to a classification problem?

10. What is the role of the bias term in the discriminant function?

11. What would be limitations imposed by the lack of a bias term in the discriminant function, i.e., what would happen if we use f(x)=w^Tx instead of f(x)=w^Tx+b?

12. Calculate the analytical form of the gradient vector of the following function (with respect to variables a and b):

f(a,b)=(a-b)^2

f(a,b)=a^2-ab

f(a,b)=a-ab

13. What is meant by convexity? Use your understanding of function convexity to determine if each of the following functions is convex or not.

f(x)=2x+3

f(x)=(x-2)^2+4

f(x)=4(x-2)^3+3

f(x)=sin(x)

14. Describe, in your own word, the gradient descent algorithm?

15. What is the role of the learning rate or step size parameter in the gradient descent algorithm? Use the provided code to understand the role of this parameter.

16. Use the provided gradient descent code to calculate the minima of each of the following functions. Verify if the answer is correct by solving using calculus (calculating the derivative and setting it to zero by hand).

f(x) = 2x^2-x-2

f(x) = sin(3x)

f(x)=2x+3

f(x)=(x-2)^2+4

f(x)=4(x-2)^3+3

f(x)=sin(x)

17. What are the limitations of the gradient descent algorithm?

18. Why does the gradient descent algorithm fail for non-convex functions? How can we remedy this issue?

19. What is the 0-1 loss? What are the problems associated with the 0-1 loss function?

20. What is the hinge loss function?

21. What is the perceptron algorithm? What are the representation, evaluation and optimization components of the perceptron?

22. Use the provided perceptron code to understand the role of the learning rate parameter by solving simple AND, OR and XOR classification problems.

23. What is the effect of the choice of initial weights and biases in the perceptron code? Modify the code and try w

24. Use the perceptron algorithm to solve the classification exercises given in Q.5-Q.8.

25. Use the builtin perceptron algorithm in sk-learn ([https://scikit-learn.org/stable/modules/generated/sklearn.linear\_model.Perceptron.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html)) to solve the classification exercises given in Q.5-Q.8.

26. What is the perceptron learning rule convergence theorem?

27. Do you think that the perceptron algorithm can be a universal computing machine? ([https://en.wikipedia.org/wiki/Universal\_Turing\_machine](https://en.wikipedia.org/wiki/Universal_Turing_machine)) Give your reasoning.

###


### Week-3

1. How is margin related to regularization?
2. Why is regularization needed?
3. How is structural risk minimization different from empirical risk minimization?
4. What is role of the parameter &quot;C&quot; in an SVM?
5. As discussed in the lectures, some SVM formulations use &quot;Lambda&quot; in their expression instead of &quot;C&quot;. What is the relationship between &quot;C&quot; and &quot;Lambda&quot;?
6. What is a support vector?
7. Write the weight update step from gradient descent for an SVM?
8. How can you determine if the objective function in the optimization problem underlying an SVM is convex or non-convex?
9. What are the advantages of linear discriminant based classifier such as perceptron or a support vector machine?
10. How do transformations of data points change the concept of distance in the feature space?
11. How do transformations of data points allow us to use a linear discriminant in the transformed space to solve an originally linearly non-separable classification problem?
12. What is the relationship between distance and dot products?
13. How can we achieve implicit transformations of the data by changing the definition of dot products to kernel functions?
14. What is a kernel function?
15. What is the representer theorem?
16. How does the representer theoreom allow us to use kernels in an SVM?
17. Write the discriminant function of a support vector machine using the representer theoreom?
18. Obtain an expression for the optimization problem underlying a kernelized support vector machine in terms of the data points and the representer theorem.
19. Solve the optimization problem underlying a kernelized support vector machine with respect to &quot;alpha&quot; using gradient descent algorithm by writing the update expression for alpha.
20. What is the Gram matrix?
21. What are the conditions for a function to be a valid kernel function?
22. What is the role of the degree (d) and coefficient (c) in a polynomial kernel k(a,b)=(a^t b + c)^d? What impact do these have on the classification boundary?
23. What is the role of the parameter &quot;gamma&quot; in an RBF kernel? How does it affect the classification boundary?
24. How can you specify your own kernel function in the SVM?
25. How does using a kernel matrix eliminate the need for explicit feature representation of examples for a classification problem?

### Week-4

1. What is the objective of validation?
2. What is stratified validation?
3. What are underlying assumptions for accuracy as a metric?
4. What is precision, recall, false positive rate?
5. Why are accuracy, precision, recall etc. dependent upon the threshold of the classifier?
6. How do precision, recall and false positive rate change as the threshold of the classifier is increased?
7. What is the ROC curve?
8. How does area under the ROC curve serve as a performance metric?
9. Why is the ROC curve called the ROC curve?
10. How does the performance estimate of your model change with increase in the size of your validation set?
11. What are the limitations of ROC curves?
12. What is the most important region of an ROC curve?
13. How is a precision recall curve useful?
14. What is the relationship between the ROC and Precision-Recall curves?
15. What are the limitations of the precision recall curve?
16. How do we train the final model for deployment?
17. How can you choose an &quot;operating point&quot; for a machine learning model?
18. What is the impact of the choice of K in K-fold cross validation on performance statistics?
19. What K should we use?
20. What is F1?
21. What is Matthews Correlation Coefficient?
22. Why are FPR and TPR monotonically non-increasing functions of threshold but precision is not?
23. What is grid search?
24. What is bootstrapping? What are .632 and .632+ bootstrap?

Week-5

A number of questions are included in the PCA Lecture files and the eigen decomposition and PCA tutorials linked below:

[https://github.com/foxtrotmike/PCA-Tutorial/blob/master/Eigen.ipynb](https://github.com/foxtrotmike/PCA-Tutorial/blob/master/Eigen.ipynb)

[https://github.com/foxtrotmike/PCA-Tutorial/blob/master/pca-lagrange.ipynb](https://github.com/foxtrotmike/PCA-Tutorial/blob/master/pca-lagrange.ipynb)

Week-6

What is meant by a manifold?

What is the SRM formulation of PCA?

What is the primary idea behind incremental PCA?

1. What is the fundamcental idea of robust regression?
2. What is UMAP? How does it work? What is the role of the neighborhood and distance constraint parameter for UMAP?
3. What is meant by t-SNE? How does it work?
4. How can PCA be kernelized?
5. What does &quot;Component Analysis&quot; mean in general? How is it different from &quot;Discriminant Analysis&quot;?
6. What is meant by regression? How is it different from classification?
7. Derive a closed-form formula for the optimal weights of ordinary least squares regression?
8. Can OLS be used for classification? Give a justification of why this is or is not a good choice.
9. What is the difference in terms of Representation, Evaluation and Optimization between each of the following models: Ordinary Least Squares Regression, Ridge Regression, Lasso Regression, Support Vector Regression, Logistic Regression?
10. Is Logistic Regression a regression method or a classification method? Given an explanation for this in terms of the loss function used in logistic regression?
11. What are the limitations of square error loss?
12. What is the motivation behind epsilon-insensitive loss function?
13. What is meant by pseudo-inverse of a matrix?
14. How can you improve hurricane intensity prediction?
15. How is ordinary least squares regression related to solving a system of linear equations?

Week-7

1. Write the represenation and evaluation of different types of loss functions used in various problems discussed in this week.
2. What are the desired characteristics of loss function for a given problem?
3. What is the difference between L0, L1 and L2 regularization?
4. What is the impact of L0, L1 and L2 regularization?
5. What is the main idea behind: ranking, reinforcement learning, survival prediction, etc.
6. What are the performance metrics for each type of machine learning problem discussed in this week?
7. How can you use click data to obtain training data in the design of recommendation systems?
8. Why is collaborative filtering called &quot;collaborative&quot; filtering?
9. How can you use cross-validation and other performance assessment techniques for different types of ML problems discussed in this week?
10. Can you describe the process of obtaining solution to a novel machine learning problem?

Week-8

1. What is meant by a neuron, soma, axon, dendrites, synaptic gap?
2. What is meant by &quot;firing&quot; of a neuron?
3. What is the mathematical model of a neuron?
4. What is meant by activation function?
5. What is the role of the activation function?
6. What is a neural network?
7. What is meant by a fully connected feed-forward neural network?
8. Write the representation of a fully connected neural network in mathematical form for an input x?
9. What is the evaluation of a FCNN?
10. How can we optimize a neural network?
11. What is meant by a layer of neurons?
12. What is the impact of adding neurons to a neural layer?
13. What is the impact of adding more layers?
14. Would a neural network with multiple layers and multiple neurons in each layer be able to classify a dataset that is not linearly separable?


Week-9

1. What is universal approximation?
2. How are deeper models more compact representations?
3. What is the effect of adding layers in a neural network?
4. What is the impact of adding more neurons in a layer in a neural network?
5. What are the problems with using a fully connected neural network or a Multi-layered perceptron for image detection and classification?
6. What is convolution?
7. What is a filter?
8. How is the difference between convolution and correlation?
9. How can convolution be viewed as a dot product?
10. What is a CNN? What are its major components?
11. What is Pooling?
12. What is stride?
13. what is padding?
14. What is meant by a receptive field?
15. How can you calculate the number of features at the output of convolutional blocks for flattening in a CNN?
16. What is weight decay?
17. How is regularization achieved in CNNs?
18. What is drop out?
19. What is batch normalization?
20. What is transfer learning?

Week-10

1. What is a residual network layer?
2. Why is a resnet better at modeling?
3. What is generative modeling?
4. What is the REO for autoencoders?
5. What is the fundamental idea behind GANs?
6. What is stop-word remove in NLP applications?
7. What is stemming?
8. What is leammtization?
9. What is meant by a dictionary for a corpus?
10. What is meant by term-frequency?
11. What is TF-IDF?
12. How can you solve a document classification with TF-IDF?
13. What is the fundamental idea behind language models such as GPT-2 and BERT?
