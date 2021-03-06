# Wildflower Finder
[wildflowerfinder.com](http://wildflowerfinder.com)

## Image Classification from Mobile Phone Images with Deep Learning

![](https://cloud.githubusercontent.com/assets/17363251/26757751/d79740ea-4884-11e7-8c55-51cadfe08fb4.jpg)
<sup>Images &copy; Jennifer Waller 2017</sup>

This project uses convolutional neural nets to classify images of wildflowers found in Colorado's Front Range.

## Motivation
Accurate identification of wildflowers is a task with relevance to both recreation and environmental management. Currently, there are several mobile apps designed to identify flowers using images; the best of these (e.g., [https://identify.plantnet-project.org/](https://identify.plantnet-project.org/)) is connected with an annual international competition for advancing techniques for plant classification from images. However, none of the extant plant identification apps are particularly accurate for identification of flowers in North America.

Example: Pl@ntNet attempt to identify Delphinium nuttalianum, a plant commonly seen blooming in Colorado in early spring:

![](https://user-images.githubusercontent.com/17363251/27239801-a2692752-528f-11e7-9625-ba15fa810135.jpg)

<!-- As of 6/15/17, iNaturalist announced plans to release image classification as part of their nature reporting app ([https://www.inaturalist.org/](https://www.inaturalist.org/)) -->

It seems reasonable that a model trained primarily on images of flora prevalent in the Front Range of Colorado would be more likely to correctly identify images of local wildflowers than global apps trained on flora located primarily in other regions of the world. The primary aim of this project is to develop a model for classification of wildflowers native to the Front Range in Colorado. A secondary aim is to develop a model that, in future, could take advantage of metadata provided by users of a mobile app while photographing wildflowers in order to provide more accurate classifications.

## Data
Initially, I planned to collect images via web scraping. However, my preliminary efforts suggested that web scraping would be very time intensive as most websites with images of wildflowers have only a few images of each species. Additionally, when considering ways to improve upon existing flower identification apps, it seemed to me that having photographs tagged with date/time and GPS location could be potentially useful. In the long term, historical GPS and date/time information could be used to improve prediction of flower species; each species is more common in particular areas/elevations and at particular times of the year. More immediately, GPS information will permit clustering of photos by location, which will allow me to cluster images within observations (i.e., one plant = one observation), a strategy employed in the 2015 LifeCLEF challenge (for a summary, see [http://ceur-ws.org/Vol-1391/157-CR.pdf](http://ceur-ws.org/Vol-1391/157-CR.pdf)). For all these reasons, I chose to collect photographs of local wildflowers using my iPhone and a point and shoot camera. I also gathered mobile phone photos from friends and family.

## Neural Nets

### Baseline Model

Basic hand-written CNN using Keras with Theano backend, trained on photos taken with my iPhone 6s.
* <b>Data</b>: For this model, I was pickier with images than in later attempts (see below); I only included images that were in focus and I removed images that were very similar.
    * 651 images representing 11 wildflower species/classes
    * Images resized to 120 x 90

* <b>Results</b>: Accuracy was .88. Misclassified images were most commonly images confused as penstemon virens (suggesting that I needed more photos of penstemon virens) or images with a lot of foliage. This seemed to be due to the relative infrequency of zoomed-out images containing a lot of foliage within the data set, generally. To resolve this issue, I considered adding more zoomed-out images or simply using higher resolution images or cropping the images. The foliage-related misclassification issue is demonstrated by the images below:

    ![](https://user-images.githubusercontent.com/17363251/27846464-bc9be920-60f4-11e7-9f1e-d6066ffebade.png)

* <b>Next Steps</b>: A brief review of the literature related to image classification for flowers brought me to publications from recent successful teams in the PlantCLEF ([http://www.imageclef.org/lifeclef/2016/plant](http://www.imageclef.org/lifeclef/2016/plant)) annual competition. I was particularly interested in the possibility of using a deep residual network based on work from Šulc and colleagues ([http://cmp.felk.cvut.cz/~mishkdmy/papers/CMP-CLEF-2016.pdf](http://cmp.felk.cvut.cz/~mishkdmy/papers/CMP-CLEF-2016.pdf)).  

### ResNet50

The current standard for plant identification is fine tuning very deep networks trained on large datasets of images (e.g., ImageNet ([http://www.image-net.org/](http://www.image-net.org/))). One of the newer advances in deep networks is He and colleagues' residual neural network, ResNet ([https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385))). Deep networks have been of great interest to computer vision researchers because neural networks with more layers are able to recognize more features than those with fewer layers. Being able to recognize more features is very useful for differentiating objects with a lot of visual complexity, like flowers. However, traditional neural networks suffer from oversaturation when they have a lot of layers; they actually <i>underfit</i> on the training data. Residual networks differ from 'traditional' deep networks because the model is trained to learn the residual error instead of the traditional mapping. ResNet also passes the identity mapping past convolutional layers in parts of the model; this also reduces the chance of oversaturation.

![](https://user-images.githubusercontent.com/17363251/27404074-fd52f01e-5689-11e7-9a5b-52705745c26f.png)

Image from He et al., 2015 paper:  [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)

* <b>Fine-tuning of pre-trained ResNet50</b> (Keras build from [https://github.com/fchollet/keras/blob/master/keras/applications/resnet50.py](https://github.com/fchollet/keras/blob/master/keras/applications/resnet50.py)). ResNet50 was trained on millions of images of objects, so it is already trained to detect basic features in objects (e.g., edges, colors). By adding fully connected layers specific to the wildflower data, we essentially fine tune ResNet50 to apply its understanding of basic objects to identify features that distinguish our flower species/classes.

    * <b>Base Model</b> = ResNet50 trained on Imagenet dataset
    * <b>Fully connected layers</b> are specific to this project:
        1. Flatten
        2. Dense (activation = relu)
        3. Dense (matches shape of 13 flower classes, activation=softmax)
        4. Compiling model:
            * Optimizer = SGD
            * Loss = categorical crossentropy

* <b>Image Preprocessing</b>: This time, I wanted to try using image generation to reduce overfitting (see below). To do this, I first needed to resize (to 256x256) and center/crop (to 224x224) the images.

    ![](https://user-images.githubusercontent.com/17363251/26950899-86a595f2-4c5c-11e7-9de0-a60f0d66200c.png)

* <b>Image Generation</b>: To decrease the chance of overfitting, the image generator in Keras provided augmented images for each epoch; thus, the model never saw same image twice. Random augmentations included horizontal flip, rotation (up to 30 degrees), horizontal and vertical shift.

![](https://user-images.githubusercontent.com/17363251/26950488-04433fc0-4c5b-11e7-8746-2f0fe0c5f13a.jpg)

* <b>Data</b>: Using all available data (including out of focus photos):
    * 1,526 images / 13 species of flowers


![](https://user-images.githubusercontent.com/17363251/27403825-1685cdd2-5689-11e7-9e4f-0eacba2e0c9c.jpg)

* <b>Training the Model</b>
    * Set aside 20% of data (<i>n</i> = 306) for validation data set
    * Trained model with train/test split (80% train) of remaining images (<i>n</i> = 1,220) on NVIDIA GPU using Amazon AWS EC2 instance.

![](https://user-images.githubusercontent.com/17363251/27237307-dfb1768c-5285-11e7-8986-8b2455a2a988.png)

## Results

* Accuracy with random guessing, given the class imbalance, would be .09.
* Model accuracy on validation data: .97
    * 97% accuracy is pretty good!
    * Only misclassified 4/306 flowers

![](https://user-images.githubusercontent.com/17363251/27405111-8bbdf6d4-568d-11e7-9066-4d97988cb0be.png)

* The misclassified images look like they were challenging cases (i.e., side views (rare), blurred images, unusual bloom appearance for a given class).

## See WildflowerFinder in Action!

[wildflowerfinder.com](http://wildflowerfinder.com)

<!-- Concerned that using all data leaves in some images that are very similar (e.g., when I tried repeatedly to take a nice shot and thus have 2 or 3 very similar images), so removed images that were very similar.  -->


## Future Directions

* Add more classes/more images
* Include images from cameras other than my iPhone 6
* Bagging of multiple deep networks to improve accuracy with more classes
* Object recognition: automated cropping
* Cluster images by geotags


<!-- ### Geotagged Images

I hoped to be able to use gps location to improve model accuracy by allowing 'voting' on species classification by images taken for the same plant instance. This requires first labeling images that were taken of the same plant as belonging together. (See [exif_gps.py](https://github.com/jw15/capstone/blob/master/src/exif_gps.py) for code.) Unfortunately, when I used a third party camera app to take plant images, the app saved the location where I saved all the images to my iPhone's camera roll as the gps tag for every images. Thus, the GPS information for those images is not usable. However, I do have many images taken with my iPhone native phone app and these do have correct GPS tags. Another potential issue was accuracy/sensitivity of the GPS tags provided by the iPhone; fortunately, the GPS tags from iPhone's native camera app seem to be sufficiently sensitive for identifying individual plants. -->

<!-- This is a plot showing GPS locations for two plant species (achillea lanulosa, sand lily): [(plot)](http://ec2-34-226-23-205.compute-1.amazonaws.com:8105/#) -->

## Tools



![](https://user-images.githubusercontent.com/17363251/27443877-b73a3378-5731-11e7-87b8-643925a8c7b2.png)




## References
<!--
Dodge, S., & Karam, L. (2016). Understanding how image quality affects deep neural networks. [(https://arxiv.org/pdf/1604.04004.pdf)](https://arxiv.org/pdf/1604.04004.pdf) -->
<!--
Jaderberg, M., Simonyan, K., Zisserman, A., & Kavukcuoglu, K. (2016). Spatial transformer networks. [(https://arxiv.org/pdf/1506.02025.pdf)](https://arxiv.org/pdf/1506.02025.pdf) -->
Goëau, H., Bonnet, P., & Joly, A. (2015). LifeCLEF Plant Identification Task 2015. [(http://ceur-ws.org/Vol-1391/157-CR.pdf)](http://ceur-ws.org/Vol-1391/157-CR.pdf)

He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep residual learning for image recognition. arXiv.org. [(https://arxiv.org/abs/1512.03385)](https://arxiv.org/abs/1512.03385)

<!-- Simonyan, K., & Zisserman, A. (2015). Very deep convolutional networks for large­scale image recognition. Published as conference paper at ICLR. [(https://arxiv.org/pdf/1409.1556.pdf)](https://arxiv.org/pdf/1409.1556.pdf)
Package: [(https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3)](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3) -->

 Šulc, M., Mishkin, D., & Matas, J. (2016). Very deep residual networks with MaxOut for plant identification in the wild. [(http://cmp.felk.cvut.cz/~mishkdmy/papers/CMP-CLEF-2016.pdf)](http://cmp.felk.cvut.cz/~mishkdmy/papers/CMP-CLEF-2016.pdf)

 &copy; Jennifer Waller 2017. All rights reserved.
