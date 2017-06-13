
'''
QUESTIONS:
* Why are the number of observations in my net with ResNet == 44 when I know there are over 1300 images in x_train?
    * ANSWER: 44 is the number of batches (26 images per batch).
* How can I do bagging on CNNs? What would that mean?
    * Run series of models with different bootstrapped sample each time. Run test data through all models and vote/average to get final prediction?


THINGS TO TRY:
* stratify validation data so one class isn't over-represented there.
    * done. Stratified max validation accuracy = .9921. BEFORE stratifying, max accuracy was .997. So, stratifying didn't improve fit (for seed=1337)
* Visualize layers in neural net
* Plot model loss/accuracy ('learning curve') over time
    * Should be working now.
* Model seems to plateau at about 26 epochs; adjust training rate (divide by 10) at that point?
    * Added function to adjust training rate when it plateaus.


From Frank (topics to address in presentation):
how is cnn different from mlp?
why choose different activation functions
regularization

sklearn kfolds to create masks from indices
    Looked into this and it seems like bagging is better alternative for cnns.

Random guessing accuracy using proportion of classes/representation of classes (how many in each class in training set?)

Get most representative image for each class.

COULD write package to return f1 score, precision, recall for keras.
'''
