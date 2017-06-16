# Wildflower Finder
### Image Classification from Mobile Phone Images

![](https://cloud.githubusercontent.com/assets/17363251/26757751/d79740ea-4884-11e7-8c55-51cadfe08fb4.jpg)
<sup>Images &copy; Jennifer Waller 2017</sup>

This project uses convolutional neural nets to classify images of wildflowers found in Colorado's Front Range.

### Motivation
Accurate identification of wildflowers is a task with relevance to both recreation and environmental management. Currently, there are several mobile apps designed to identify flowers using images; the best of these (e.g., [https://identify.plantnet-project.org/](https://identify.plantnet-project.org/)) is connected with an annual international competition for advancing techniques for plant classification from images. However, none of the extant plant identification apps are particularly accurate for identification of flowers in North America.

Example: Pl@ntNet attempt to identify Penstemon virens:

![](https://user-images.githubusercontent.com/17363251/27238145-0c3c0d5e-5289-11e7-9b8e-fc5d1523de40.jpg)

As of 6/15/17, iNaturalist announced plans to release image classification as part of their nature reporting app ([https://www.inaturalist.org/](https://www.inaturalist.org/))

It seems reasonable that a model trained primarily on images of flora prevalent in the Front Range of Colorado would be more likely to correctly identify images of local wildflowers than global apps trained on flora located primarily in other regions of the world. The primary aim of this project is to develop a model for classification of wildflowers native to the Front Range in Colorado. A secondary aim is to develop a model that, in future, could take advantage of metadata provided by users of a mobile app while photographing wildflowers in order to provide more accurate classifications.

### Data
Initially, I planned to collect images via web scraping. However, my preliminary efforts suggested that web scraping would be very time intensive as most websites with images of wildflowers have only a few images of each species. Additionally, when considering ways to improve upon existing flower identification apps, it seemed to me that having photographs tagged with date/time and GPS location could be potentially useful. In the long term, historical GPS and date/time information could be used to improve prediction of flower species; each species is more common in particular areas/elevations and at particular times of the year. More immediately, GPS information will permit clustering of photos by location, which will allow me to cluster images within observations (i.e., one plant = one observation), a strategy employed in the 2015 LifeCLEF challenge (for a summary, see [http://ceur-ws.org/Vol-1391/157-CR.pdf](http://ceur-ws.org/Vol-1391/157-CR.pdf)). For all these reasons, I chose to collect photographs of local wildflowers using my iPhone and a point and shoot camera. I also gathered mobile phone photos from friends and family.

### Neural Nets

#### Baseline Model

* Basic CNN using Keras, trained on 651 categorized and in­-focus photos, taken on my iPhone 6s, representing 11 local wildflower species. Included only in-focus images. Removed images that were very similar. Images resized to 120 x 90. Accuracy was .88. The misclassified images suggest that I need more photos of one frequently misclassified species (i.e., penstemon virens) and may need to run a model using higher resolution images or consider cropping images. The latter issue is demonstrated by the images in Figure 1.

    ![](https://cloud.githubusercontent.com/assets/17363251/26746371/55be1a22-47ac-11e7-97c7-4fb6e1cebfa2.png)

#### ResNet50

* Current standard for plant identification is fine tuning very deep networks trained on large datasets of images (e.g., Imagenet). One of the newest advances in deep networks is residual neural networks (i.e., ResNet). These differ from 'traditional' deep networks because the model is trained to learn the residual error instead of the traditional mapping. This prevents oversaturation of the model (and overfitting on training data with resulting accuracy loss on validation data) that results from having a lot of layers in a more traditional CNN.

<img src="https://user-images.githubusercontent.com/17363251/27237890-048256dc-5288-11e7-9bf7-22187d02f122.png" alt="" width="300" height="500">

![](https://user-images.githubusercontent.com/17363251/27237890-048256dc-5288-11e7-9bf7-22187d02f122.png){:class="img-responsive"}
![](https://user-images.githubusercontent.com/17363251/27237890-048256dc-5288-11e7-9bf7-22187d02f122.png){:height="50%" width="50%"}
![](https://user-images.githubusercontent.com/17363251/27237890-048256dc-5288-11e7-9bf7-22187d02f122.png){:height="700px" width="400px"}
Image from [http://felixlaumon.github.io/2015/01/08/kaggle-right-whale.html](http://felixlaumon.github.io/2015/01/08/kaggle-right-whale.html)

* Fine-tuning of pre-trained ResNet50 (Keras build from [https://github.com/fchollet/keras/blob/master/keras/applications/resnet50.py](https://github.com/fchollet/keras/blob/master/keras/applications/resnet50.py)), trained on 970 photos representing 13 species. ResNet50 was trained on the Imagenet data, containing millions of images of objects.

    * Base model = ResNet50 trained on Imagenet dataset
    * Fully connnected layers are specific to this project;
        1. Flatten
        2. Dense (activation = relu)
        3. Dense (matches shape of 13 flower classes, activation=softmax)
        4. Compiling model:
            * Optimizer = SGD
            * Loss = categorical crossentropy

* Image Preprocessing: Resize (to 256x256), center/crop (to 224x224), and normalize images

    ![](https://user-images.githubusercontent.com/17363251/26950899-86a595f2-4c5c-11e7-9de0-a60f0d66200c.png)

* Image generation: To decrease the chance of overfitting, the image generator in Keras provided augmented images for each epoch; thus, the model never saw same image twice. Random augmentations included horizontal flip, rotation (up to 30 degrees), horizontal and vertical shift.

![](https://user-images.githubusercontent.com/17363251/26950488-04433fc0-4c5b-11e7-8746-2f0fe0c5f13a.jpg)

Using all data (not only 'nice' shots):

* 1,526 images / 13 types of flowers:

| Flower Species | Images |
| ---------------------------------- | ------- |
| Achillea lanulosa| 117 |
| Adenolinum lewissi | 184 |
| Alyssum parviflorum | 145 |
| Arnica cordifolia | 80 |
| Cerastium arvense | 172 |
| Delphinium nuttallianum | 65 |
| Harbouria trachypleura | 73 |
| Lonicera tatarica | 132 |
| Mertensia lanceolata | 156 |
| Padus virginiana | 156 |
| Penstemon virens | 127 |
| Phlox longifolia | 71 |
| Senecio integerrimus | 48 |

* Set aside 20% of data (n = 306) for validation data set
* Trained model with train/test split (80% train) of remaining images (n = 1,220)
    * Model accuracy on validation data: .97
    * Misclassified 4/306 flowers
    * Accuracy with random guessing, given the class imbalance, would be .09.

![](https://user-images.githubusercontent.com/17363251/27237307-dfb1768c-5285-11e7-8986-8b2455a2a988.png)

<!-- Concerned that using all data leaves in some images that are very similar (e.g., when I tried repeatedly to take a nice shot and thus have 2 or 3 very similar images), so removed images that were very similar.  -->


### Future Directions:

* Web App!
* Include images from cameras other than my iPhone 6.
* Experiment with other pre-trained deep learning models.
* Try bagging of multiple deep networks.
* Object recognition
* Cluster images by geotags?


<!-- ### Geotagged Images

I hoped to be able to use gps location to improve model accuracy by allowing 'voting' on species classification by images taken for the same plant instance. This requires first labeling images that were taken of the same plant as belonging together. (See [exif_gps.py](https://github.com/jw15/capstone/blob/master/src/exif_gps.py) for code.) Unfortunately, when I used a third party camera app to take plant images, the app saved the location where I saved all the images to my iPhone's camera roll as the gps tag for every images. Thus, the GPS information for those images is not usable. However, I do have many images taken with my iPhone native phone app and these do have correct GPS tags. Another potential issue was accuracy/sensitivity of the GPS tags provided by the iPhone; fortunately, the GPS tags from iPhone's native camera app seem to be sufficiently sensitive for identifying individual plants. -->

<!-- This is a plot showing GPS locations for two plant species (achillea lanulosa, sand lily): [(plot)](http://ec2-34-226-23-205.compute-1.amazonaws.com:8105/#) -->

### Tools

Python (Numpy, Pandas, Keras, Theano, Scikit­-Learn, OpenCV, PIL, SciKit-­Image, Flask, possibly BeautifulSoup), AWS (EC2, S3), ImageMagick

### References

Dodge, S., & Karam, L. (2016). Understanding how image quality affects deep neural networks. [(https://arxiv.org/pdf/1604.04004.pdf)](https://arxiv.org/pdf/1604.04004.pdf)

Jaderberg, M., Simonyan, K., Zisserman, A., & Kavukcuoglu, K. (2016). Spatial transformer networks. [(https://arxiv.org/pdf/1506.02025.pdf)](https://arxiv.org/pdf/1506.02025.pdf)

Simonyan, K., & Zisserman, A. (2015). Very deep convolutional networks for large­scale image recognition. Published as conference paper at ICLR. [(https://arxiv.org/pdf/1409.1556.pdf)](https://arxiv.org/pdf/1409.1556.pdf)
Package: [(https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3)](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3)
