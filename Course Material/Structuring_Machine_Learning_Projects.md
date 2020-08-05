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

The ceiling of the problem is how much you can reasonably expect something to improve if a change is made. Should I make my classifier work better for misclassifying? Depends. Take 100 mislabeled examples and count how many out of those 100 are misclassified and see if that'll make a huge difference in the end.  YOu can also evaluate multiple ideas in parallel using this method. You create a table, with the images on the left, and the columns correspond to the problem or miscategorized label. Then mark the percentage of images have checkmarks -> how many incorrect mislabeling is causing problems. 

Sometimes you can have data that's incorrectly labeled. Human labeling can actually be incorrect. DL algorithms are robust to random errors in the training set. However, systematic errors are another issue. Counting incorrectly labeled labels and seeing how much of a problem it is is a good idea. Look at the overall dev set area, errors due to incorrect labels, and errors due to other causes. Some guidelines are applying the process of correction to both dev and test sets to keep continuity. Also consider examining examples your algorithm got right as well. 

Build the first system quickly and then iterate. Set up a dev/test set metric and build the initial system quickly. You can use bias/variance analysis to prioritize next steps. 

### Mismatched Training and Dev/Test Set

More teams are training on data that are different with training/testing on different distributions. One thing you can do is combine the datasets and randomly shuffle them into a train/dev/test set. The advantage is the same distribution but the disadvantage is that a lot of data come from a place where you don't care for. Dev set is where you're aiming the target. The other, recommended option, is having the training set have half mobile app and web data; the test/dev being all from the mobile application. 

Ways you analyze bias/variance changes when you have mismatched data distributions. You look at training and dev error usually. But how much variance is due to the algorithm versus due to mismatched data is difficult to tease out. Defining a new piece of data called training-dev set (same distribution as training set but not used for training). This can help you tease out if variance is the issue. Data mismatch when the training and train-dev are close but the error with dev and training are high. 

If you have a mismatch problem, there aren't many systematic solutions. Carry out manual error anlysis to try to understand difference between training and dev/test sets. Make the training data more similar or collect data more similar to the dev/test sets. There's also artificial data synthesis that can be utilized. There's a risk that overfitting could occur though if you don't do this process properly. 

### Learning from Multiple Tasks 

You can take knowledge from one network and apply it for other tasks (transfer learning). You swap in a new dataset, initialize the last layer's weights randomly, and then retrain. These things are referring to pretraining and fine-tuning. This helps with learning faster or with less data. You can also create new layers on top of a new output, that may be pretty useful. This is useful when you don't have much data. This is a good system to use when you have a ton of data for the initial machine learning model. 

Multi-task learning is when you can have simultaneous tasks with one neural network doing several things simultaneously. One image can have multiple labels. The benefit is that you can train one neural network to basically accomplish the work of four neural networks. Multi-task learning makes sense when training on a set of tasks that could benefit from sharing lower level features and when the amount of data for each task is quite similar. Finally, when you can train a big enough neural newtork to do well on all the tasks. 

### End to End Deep Learning

Typically, you would extract features and then combine the individual parts to make an output. With end-to-end deep learning, you can train a neural network to do a lot of the intermediate tasks. You might need a ton of data before end to end deep learning works really well. Breaking problems to two steps, cropping the image and feeding it to a neural network. It's also a data issue to solve subproblems. End-to-end learning lets the learning speak for itself -> you can force machines to remove pre-conceived bias in the algorithm. Less design work is necessary. However, the costs are large amounts of data. Disadvantage is removing potentially useful hand-designed features. 


