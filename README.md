# Colorado Wildflower Classification from Mobile Phone Images

![](https://cloud.githubusercontent.com/assets/17363251/26757751/d79740ea-4884-11e7-8c55-51cadfe08fb4.jpg)
<sup>Images &copy; Jennifer Waller 2017</sup>

This project uses convolutional neural nets to classify images of wildflowers found in Colorado's Front Range.

### Motivation
Accurate identification of wildflowers is a task with relevance to both recreation and environmental management. Currently, there are several mobile apps designed to identify flowers using images; however, none of these apps are particularly accurate for identification of flowers in North America, probably because they were trained primarily on species prevalent in Europe. It seems reasonable that a model trained primarily on images of flora prevalent in the Front Range of Colorado would be more likely to correctly identify images of local wildflowers than global apps trained on flora located primarily in other regions of the world. The primary aim of this project is to develop a model for classification of wildflowers native to the Front Range in Colorado. A secondary aim is to develop a model that, in future, could take advantage of metadata provided by users of a mobile app while photographing wildflowers in order to provide more accurate classifications.

### Data
Initially, I planned to collect images via web scraping. However, my preliminary efforts suggested that web scraping would be very time intensive as most websites with images of wildflowers have only a few images of each species. Additionally, when considering ways to improve upon existing flower identification apps, it seemed to me that having photographs tagged with date/time and GPS location could be potentially useful. In the long term, historical GPS and date/time information could be used to improve prediction of flower species; each species is more common in particular areas/elevations and at particular times of the year. More immediately, GPS information will permit clustering of photos by location, which will allow me to cluster images within observations (i.e., one plant = one observation), a strategy employed in the 2015 LifeCLEF challenge (for a summary, see [http://ceur-ws.org/Vol-1391/157-CR.pdf](http://ceur-ws.org/Vol-1391/157-CR.pdf)). For all these reasons, I chose to collect photographs of local wildflowers using my iPhone and a point and shoot camera. I also gathered mobile phone photos from friends and family.

To date, I have successfully trained a very basic convolutional neural net using Keras on 651 categorized and in­-focus photos, taken on my iPhone 6s, representing 11 local wildflower species. The misclassified images suggest that I need more photos of one frequently misclassified species (i.e., penstemon virens) and may need to run a model using higher resolution images or consider cropping images. The latter issue is demonstrated by the images in Figure 1.

![](https://cloud.githubusercontent.com/assets/17363251/26746371/55be1a22-47ac-11e7-97c7-4fb6e1cebfa2.png)

I also have several hundred additional photos representing ‘challenging cases’ (i.e., photos that are less well focused, are side views, or that feature a less clear image of the flower) and flower classes (10 species) that currently have insufficient numbers of images to include in the model. I plan to collect many more images over the next few weeks. There is  some class imbalance in the data that I hope to correct by collecting more images; I also plan to include more of those underrepresented species in the model.

### Data Project

1. Collect photographs of Colorado wildflowers with my iPhone, point and shoot camera, and others’ iPhones. Possibly supplement them with images scraped from websites, such as Bing, [easterncoloradowildflowers.com](www.easterncoloradowildflowers.com), etc.

2. Resize (to 256x256), center/crop (to 224x224), and normalize images

    ![](https://user-images.githubusercontent.com/17363251/26950899-86a595f2-4c5c-11e7-9de0-a60f0d66200c.png)

2. Image generation: To decrease the chance of overfitting, the image generator in Keras provided augmented images for each epoch; thus, the model never saw same image twice. Random augmentations included horizontal flip, rotation (up to 30 degrees), horizontal and vertical shift.

    ![](https://user-images.githubusercontent.com/17363251/26950488-04433fc0-4c5b-11e7-8746-2f0fe0c5f13a.jpg)

3. Apply convolutional neural network for image classification
    * Built model on pre-trained ResNet50.

4. Build a web app to serve as precursor to a mobile app. Web app will accept images of wildflowers and provide classification outcome and predicted probability, information about matched flower species, image of matched species.

5. Future Directions:

    * Experiment with training models using both high and low resolution images. Existing research suggests that high resolution images may be helpful for identifying some challenging features in networks, although training on lower resolution images is likely to produce a model that is better at classifying other less­ than ­perfect images (e.g., Dodge & Karam, 2016).
    * Include images from cameras other than my iPhone 6.
    * Experiment with other pre-trained deep learning models
    * Try bagging of multiple deep networks.
    * Experiment with adding spatial transformer to first layer of network (e.g., Jaderberg et al., 2016)
    * Possibly experiment with using video or multiple photos of a single flower to produce 3D images and train a neural net on those.

6. Tools: Python (Numpy, Pandas, Keras, Theano, Scikit­-Learn, OpenCV, PIL, SciKit-­Image, Flask, possibly BeautifulSoup), AWS (EC2, S3), ImageMagick

### Geotagged Images

I hoped to be able to use gps location to improve model accuracy by allowing 'voting' on species classification by images taken for the same plant instance. This requires first labeling images that were taken of the same plant as belonging together. (See [exif_gps.py](https://github.com/jw15/capstone/blob/master/src/exif_gps.py) for code.) Unfortunately, when I used a third party camera app to take plant images, the app saved the location where I saved all the images to my iPhone's camera roll as the gps tag for every images. Thus, the GPS information for those images is not usable. However, I do have many images taken with my iPhone native phone app and these do have correct GPS tags. Another potential issue was accuracy/sensitivity of the GPS tags provided by the iPhone; fortunately, the GPS tags from iPhone's native camera app seem to be sufficiently sensitive for identifying individual plants.

This is a plot showing GPS locations for two plant species (achillea lanulosa, sand lily): [(plot)](http://ec2-34-226-23-205.compute-1.amazonaws.com:8105/#)


### References

Dodge, S., & Karam, L. (2016). Understanding how image quality affects deep neural networks. [(https://arxiv.org/pdf/1604.04004.pdf)](https://arxiv.org/pdf/1604.04004.pdf)

Jaderberg, M., Simonyan, K., Zisserman, A., & Kavukcuoglu, K. (2016). Spatial transformer networks. [(https://arxiv.org/pdf/1506.02025.pdf)](https://arxiv.org/pdf/1506.02025.pdf)

Simonyan, K., & Zisserman, A. (2015). Very deep convolutional networks for large­scale image recognition. Published as conference paper at ICLR. [(https://arxiv.org/pdf/1409.1556.pdf)](https://arxiv.org/pdf/1409.1556.pdf)
Package: [(https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3)](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3)
