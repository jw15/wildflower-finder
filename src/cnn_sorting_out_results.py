'''
Figuring out which images are misclassified by cnn
'''

ypred_indices = np.argmax(ypred, axis=1)
yactual_indices = np.argmax(Y_test, axis=1)

# Get indices for images that were misclassified
misclassified_idxs = np.where(yactual_indices - ypred_indices !=0)

for img in misclassified_idxs[0]:
    plt.imshow(X_test[img])
    classification = ypred_indices[img]
    plt.savefig('mis_idx_{}_classifiedas_{}'.format(img, classification))
