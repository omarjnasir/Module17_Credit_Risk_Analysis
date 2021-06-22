# Module17_Credit_Risk_Analysis
Module 17- README
SUPERVISED MACHINE LEARNING AND CREDIT RISK 
Overview:
Millions of people borrow money, and most contracts are performed according to payment schedule, but some may default. The purpose of this exercise is to determine credit risk. What type of borrowers are likely to default and are the lending criterion serving the interests of the lender?
Statistical Observations are made, and the data evaluated for risk. Machine Learning is used to do predictive analysis. In machine learning data is split into training and test modes and then the effectiveness of models measured as it is compared with actual results.
ANALYSIS:
•	Balanced Accuracy Score
•	*Balanced accuracy is a metric that one can use when evaluating how good a binary classifier is. It is especially useful when the classes are imbalanced, i.e., one of the two classes appears a lot more often than the other. This happens often in many settings such as anomaly detection and the presence of a disease.
•	As with all discussions on the performance of a binary classifier, we start with a confusion matrix:
 
Confusion Matrix is similar to the statistical concept (Ref.4) of Type I and Type II errors in Hypothesis Testing.

Balanced accuracy is based on two more commonly used metrics: sensitivity (also known as true positive rate or recall) and specificity (also known as true negative rate, or 1 – false positive rate). Sensitivity answers the question: “How many of the positive cases did I detect?” Or to put it in a manufacturing setting: “How many (truly) defective products did I manage to recall?” Specificity answers that same question but for the negative cases. Here are the formulas for sensitivity and specificity in terms of the confusion matrix:
 
Balanced accuracy is simply the arithmetic mean of the two:
 
For the confusion matrix, each Row in the Confusion Matrix represents an actual class, and each column represents a predicted class. This confusion matrix is also in our simulations, which are accompanying this README.

array ([[   71,   30],
       [ 2153, 14951]], dtype=int64)
*Taken from simulation.

Let us calculate:

Sensitivity = 71/71+2153 = 0.0319 or 3.19 %
Specificity =14951/14951+30 =0.9979

Therefore, balanced accuracy = 0.0319+0.9979/2=0.5149



•	Precision and Recall Scores:
Sometimes, one may want to know accuracy of positive predictions, which is called the precision of the classifier.
Precision = True Positive/True Positive + False Positive. So, if false positives are zero, Precision would be 1 or 100%. 
Recall is also called Sensitivity.

Recall= True Positives/True Positives + False Negative

The F1 score, also called the harmonic mean, can be characterized as a single summary statistic of precision and sensitivity. The formula for the F1 score is the following:
2(Precision * Sensitivity)/ (Precision + Sensitivity)
=0.0868
Balanced accuracy is simply the arithmetic mean of the two:

 
Ensemble Learning 
The concept of ensemble learning is the process of combining multiple models, like decision tree algorithms, to help improve the accuracy and robustness, as well as decrease variance of the model, and therefore increase the overall performance of the model. So weak learners are combined with others to make learning strong.
Ensemble learning builds on the idea that two is better than one. A single tree may be prone to errors, but many of them can be combined to form a stronger model. A random forest model, for example, combines many decision trees into a forest of trees.
Bootstrap aggregation, or "bagging," is an ensemble learning technique that combines weak learners into a strong learner as in the random forest model.
Boosting is another technique to combine weak learners into a strong learner. However, there is a major difference between bagging and boosting. In bagging, as you have seen, multiple weak learners are combined at the same time to arrive at a combined result.

In boosting, however, the weak learners are not combined at the same time. Instead, they are used sequentially, as one model learns from the mistakes of the previous model.
Adaptive Boosting
The idea behind Adaptive Boosting, called AdaBoost, is easy to understand. In AdaBoost, a model is trained then evaluated. After evaluating the errors of the first model, another model is trained. This time, however, the model gives extra weight to the errors from the previous model. The purpose of this weighting is to minimize similar errors in subsequent models. Then, the errors from the second model are given extra weight for the third model. This process is repeated until the error rate is minimized:
Gradient boosting, like AdaBoost, is an ensemble method that works sequentially. In contrast to AdaBoost, gradient boosting does not seek to minimize errors by adjusting the weight of the errors. 
Of the learning rates used, 0.5 yields the best accuracy score for the testing set and a high accuracy score for the training set. This is the value we'll implement in the final model. Also, note that the testing accuracy is more important here than the training accuracy.
Class imbalance refers to a situation in which the existing classes in a dataset aren't equally represented.                                                                              
 Random Oversampling
In random oversampling, instances of the minority class are randomly selected and added to the training set until the majority and minority classes are balanced. Synthetic Minority Oversampling Technique
The synthetic minority oversampling technique (SMOTE) is another oversampling approach to deal with unbalanced datasets. In SMOTE, like random oversampling, the size of the minority is increased. The key difference between the two lies in how the minority class is increased in size. As we have seen, in random oversampling, instances from the minority class are randomly selected and added to the minority class. In SMOTE, by contrast, new instances are interpolated. That is, for an instance from the minority class, a number of its closest neighbors is chosen. Based on the values of these neighbors, new values are created.
While precision ("pre" column) and recall ("rec" column) are high for the majority class, precision is low for the minority class.
Under Sampling:
Undersampling is another technique to address class imbalance. 
Undersampling takes the opposite approach of oversampling. Instead of increasing the number of the minority class, the size of the majority class is decreased.
Cluster Centroid Undersampling
Cluster centroid undersampling is akin to SMOTE. The algorithm identifies clusters of the majority class, then generates synthetic data points, called centroids, that are representative of the clusters. The majority class is then undersampled down to the size of the minority class.
Combination of both Oversampling and Undersampling.
SMOTEENN combines the SMOTE and Edited Nearest Neighbors (ENN) algorithms. SMOTEENN is a two-step process:
1.	Oversample the minority class with SMOTE.
2.	Clean the resulting data with an under-sampling strategy. If the two nearest neighbors of a data point belong to two different classes, that data point is dropped.

SUMMARY
•	Predictive Analysis can be done with machine learning in both in loan evaluation and in investments.
•	Fraud and Non-Performing Portfolios create two classes of data: majority and minority.  Machine Learning and subsequent prediction would be in error, if majority data or minority is considered only. Therefore, two basic techniques exist, which can balance minority and majority data. They are over sampling of minority data and under sampling of majority data. The third technique is SMOTEEN combines both over and under sampling. 

•	While resampling can attempt to address imbalance, it does not guarantee better results.

Simulation Results are shown in attached files.
Recommendations:
More and balanced data can improve the learning curve, but a point may be reached that further sampling would not improve results.  The target should be between 80-100% learning effectiveness. It can be done in repeated trials, as the new learner learns from the mistakes of the previous learner.

References:
1.	www.google.com
2.	Geron, Aurelien: Hands-On Machine Learning with Scikit-Learn, Keras, and Tensor Flow
3.	Wilmot, Paul: Machine Learning: An Applied Mathematics Introduction
4.	Triola, Mario: Elementary Statistics-Second California Edition
