# Hi, Frank!

My project readme ([https://github.com/jw15/capstone/blob/master/README.md](https://github.com/jw15/capstone/blob/master/README.md)) is pretty long and I figured you wouldn't want to wade through all of it. If you do, you can see my thrilling (joking!) project website with rather sparse looking zoomable plot of image GPS data (also here: [http://ec2-34-226-23-205.compute-1.amazonaws.com:8105/#](http://ec2-34-226-23-205.compute-1.amazonaws.com:8105/#))

### Aims

1. Develop a model for classification of wildflowers native to the Front Range in Colorado using a CNN trained on mobile phone images.
2. Develop a model that, in future, could take advantage of metadata provided by users of a mobile app while photographing wildflowers in order to provide more accurate classifications.

### Work similar to my project

A summary/review of LifeCLEF 2015 entries/strategy: ([https://github.com/jw15/capstone/blob/master/references/plant_recognition/Goeau_lifeclef_2015_overview.pdf](https://github.com/jw15/capstone/blob/master/references/plant_recognition/Goeau_lifeclef_2015_overview.pdf))

There's an annual international competition for plant recognition based on images collected in the field from mobile devices called LifeCLEF ([http://www.imageclef.org/lifeclef/2017/plant](http://www.imageclef.org/lifeclef/2017/plant)). It's integrated with the Pl@ntNet app development out of France. Lots of resources from that, including a giant dataset for the 2017 competition (which just ended) that I downloaded. I think a lot of it is photos collected from the Pl@ntNet app, which is potentially pretty useful (although most of the data is from plants native to Europe).

### Method similar to my project:

LifeCLEF 2016 entry built on ResNet: ([https://github.com/jw15/capstone/blob/master/references/plant_recognition/Sulc_LifeCLEF16_slides.pdf](https://github.com/jw15/capstone/blob/master/references/plant_recognition/Sulc_LifeCLEF16_slides.pdf))

I'm not going to have time to train some giant model with ImageNet data, so I'm planning to build mine on top of an existing one (e.g., ResNet50). To the extent that I have GPS data to do this, I am planning to combine the predicted probabilities for images that all came from the same plant in a pretty simple voting scheme to more accurately classify plant instances.
