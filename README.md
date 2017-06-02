# Image Classification of Wildflowers in Colorado’s Front Range

This project uses convolutional neural nets to classify images of wildflowers found in Colorado's Front Range.

## Project Proposal

### Motivation

Accurate identification of wildflowers is a task with relevance to both recreation and environmental management. Currently, there are several mobile apps designed to identify flowers using images; however, none of these apps are particularly accurate for identification of flowers in North America, probably because they were trained primarily on species prevalent in Europe. It seems reasonable that a model trained primarily on images of flora prevalent in the Front Range of Colorado would be more likely to correctly identify images of local wildflowers than global apps trained on flora located primarily in other regions of the world. The primary aim of this project is to develop a model for classification of wildflowers native to the Front Range in Colorado. A secondary aim is to develop a model that, in future, could take advantage of metadata provided by users of a mobile app while photographing wildflowers in order to provide more accurate classifications.

### Data

Initially, I planned to collect images via web scraping. However, my preliminary efforts suggested that web scraping would be very time intensive as most websites with images of wildflowers have only a few images of each species. Additionally, when considering ways to improve upon existing flower identification apps, it seemed to me that having photographs tagged with date/time and GPS location could be potentially useful. In the long term, historical GPS and date/time information could be used to improve prediction of flower species; each species is more common in particular areas/elevations and at particular times of the year. More immediately, GPS information could permit clustering of photos by location (perhaps even clustering photos of one specific plant together). For all these reasons, I chose to collect photographs of local wildflowers using my iPhone, a point and shoot camera, and iPhones of friends/family.

To date, I have successfully trained a very basic convolutional neural net using Keras (with 88% accuracy) on 651 categorized and in­-focus photos representing 11 local wildflower species. The misclassified images suggest that I need more photos of one particular species (i.e., penstemon virens) and may need to run a model using higher resolution images (Fig. 1).

I also have several hundred additional photos representing ‘challenging cases’ (i.e., photos that are less well focused, are side views, or that feature a less clear image of the flower) and flower classes (10 species) that currently have insufficient numbers of images to include in the model. I plan to collect many more images over the next few weeks. There is  some class imbalance in the data that I hope to correct by collecting more images; I also plan to include more of those underrepresented species in the model.

![](https://cloud.githubusercontent.com/assets/17363251/26746371/55be1a22-47ac-11e7-97c7-4fb6e1cebfa2.png)

### Data Project

1. Collect photographs of Colorado wildflowers with my iPhone, point and shoot camera, and others’ iPhones. Possibly supplement them with images scraped from websites, such as Bing, easterncoloradowildflowers.com, etc.
2. Resize and normalize images
3. Apply convolutional neural network for image classification
    * Current CNN model using images resized to 120 x 90 correctly classifies approximately 9/10 of the images.
    * Experiment with training models using both high and low resolution images. Existing research suggests that high resolution images may be helpful for identifying some challenging features in networks, although training on lower resolution images is likely to produce a model that is better at classifying other less­ than ­perfect images (e.g., Dodge & Karam, 2016). Also include images from cameras other than my iPhone 6.
    * Experiment with existing deep learning models (e.g., run my model on top of VGG16 16­layer network model for Keras (Simoyan & Zisserman, 2015).
    * Experiment with adding spatial transformer to first layer of network (e.g., Jaderberg et al., 2016)
    * Possibly experiment with using video or multiple photos of a single flower to produce 3D images and train a neural net on those.
4. Utilize AWS (EC2) to complete analyses, store images on AWS (S3)
5. Build a web app to serve as precursor to a mobile app. Web app will accept images of wildflowers and provide classification outcome, information about matched flower species, image of matched species.
6. Tools: Python (Numpy, Pandas, Keras, Theano, OpenCV, Scikit­Learn, SciKit­Image, Flask, possibly BeautifulSoup), AWS (EC2, S3), probably Spark

### References

Dodge, S., & Karam, L. (2016). Understanding how image quality affects deep neural networks. [(link)](https://arxiv.org/pdf/1604.04004.pdf)

Jaderberg, M., Simonyan, K., Zisserman, A., & Kavukcuoglu, K. (2016). Spatial transformer networks. [(link)](https://arxiv.org/pdf/1506.02025.pdf)

Simonyan, K., & Zisserman, A. (2015). Very deep convolutional networks for large­scale image recognition. Published as conference paper at ICLR. [(link)](https://arxiv.org/pdf/1409.1556.pdf)
Package [here](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3)

Socher, R., Huval, B., Bhat, B., Manning, C. D., & Ng, A. Y. (2014). Convolutional­ recursive deep learning for 3D object classification. [(link)](http://bit.ly/2rQHJHV)
