import matplotlib.pyplot as plt
import numpy as np

'''
Figuring out which images are misclassified by cnn - TEST data from train/test split
'''

ypred_indices = np.argmax(ypred, axis=1)
yactual_indices = np.argmax(Y_test, axis=1)

# Get indices for images that were misclassified
misclassified_idxs = np.where(yactual_indices - ypred_indices !=0)

for img in misclassified_idxs[0]:
    plt.close('all')
    plt.imshow(X_test[img])
    classification = ypred_indices[img]
    plt.savefig('mis_idx_{}_classifiedas_{}'.format(img, classification))


# Using test (validation) data that was set aside before training
holdout_score = model124.evaluate(X_test_holdout, Y_test_holdout, verbose=0)
holdout_predicts = model124.predict(X_test_holdout, verbose=0)

ypred_holdout_indices = np.argmax(holdout_predicts, axis=1)
yactual_holdout_indices = np.argmax(Y_test_holdout, axis=1)

# Get indices for images that were misclassified
misclassified_holdout_idxs = np.where(yactual_holdout_indices - ypred_holdout_indices !=0)


for img in misclassified_holdout_idxs[0]:
    plt.close('all')
    plt.imshow(X_test_holdout[img])
    classification = ypred_holdout_indices[img]
    actual_class = yactual_holdout_indices[img]
    plt.savefig('mis_holdout_idx_{}_is_{}_class_as_{}'.format(img, actual_class, classification))
