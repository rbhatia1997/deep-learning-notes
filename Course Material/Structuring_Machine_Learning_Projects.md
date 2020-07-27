# Introduction 

This Repository contains the notes that I have been writing for the Structuring Machine Learning Project course. It'll cover my assignments and other notes so I can reference them after the course's completion. This is meant for my personal use, but I've made it public in case it's helpful to others. This is the third course. 

## Week 1

### ML Strategy

There are a lot of challenges when using deep learning examples. Orthogonalization refers to designing such that one change or aspect controls a specific functionality. Assumptions in ML are to fit training well on the cost function, then fit the dev set well on the cost function, then fit the test set well on the cost function, and finally have good performance in the real world. Early stoppping wouldn't necessarily be super orthagonal because it can effect two parameters. 

### Setting Up Your Goal 

Setting up a real number evaluation metric is useful. You can look at precision and recall; precision is what percentage of images were identified correctly of those identified of a single category (of those cat photos, what % are really cats). Recall is of all the cat images, what images were correctly recognized (what's the % correct detected). Just find an evaluation metric that combines precision and recall; in this case, this will be F1 score. It's like an average/mean of precision and recall. It's the harmonic mean, which is (2/(1/P) + (1/R)). 

You can also care about running time as well! You can combine accuracy and running time as an evaluation metric? You can choose a classifier that maximizes accuracy but subject to a running time limit (less than or equal to 100ms). Satisficing metric is like a constraint in engineering terms. 

The dev set is called the development set or the holdout/cross-validation set. You use the dev set to check various algorithms. Your test and dev sets should come from the same distribution. Randomly selecting from the different regions to create test/dev sets are useful. Reflect data you expect to get in the future and consider important to do well on. 

70/30 split to training/test set. Or 60/20/20 for train, dev, and test. However, nowdays, when you have much larger data set sizes, you could have 98% training, 1% dev, and 1% test. The test set should be big enough to give high confidence in the overall system performance. You could add a weight term to your optimization algorithms that would overweight misclassifications that are fatal to user satisfaction. 

### Comparing to Human Level Performance

Bayes optimal error is the best possible error - there's no way for any function to pass that function. It's the best function from mapping from x to y. Human performance is pretty solid in general; if your performance is below human level performance, there are tools to get your algorithm to that point. You can get labeled data form humans, do a manual error analysis, etc. 

If there's a big difference between human and training error, you want to focus on bias. Maybe consider reducing variance in the case where your training error is close to human error but not that close to the dev error. Avoidable bias is the difference between Bayes/human error and training error. 

Human level error lets us estimate Bayes error, but it should be the best/optimal cases. Being clear about your purpose is ideal. Product recommendation, online advertising, logistics, etc. are areas where ML surpasses human performance. 

Avoidable bias and variance are fundamental assumptions of supervised learning. Avoidable bias is tackled with a bigger model or using a different algorithm/longer training. You could also find better hyperparameters as well. To reduce variance, try regulariation, get more data, or use a different algorithm. 

## Week 2

### Carrying Out Error Analysis 


