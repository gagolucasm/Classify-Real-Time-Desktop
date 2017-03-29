
# Classify real time desktop and speech


[image1]: ./example.png "example"

Overview
---
Team DeepThings (Mez Gebre and I) won the Best Product Category at the Deep Learning Hackathon in San Francisco. We developed in three days a real-time system capable of identifying objects and speaking what it sees, thinking about making a useful tool for the visually impaired, as it could make navigation easier. Proof of concept on a laptop, final model running on Android.

This is only the first prototype for Windows.


The goals / steps of this project are the following:
---

* Get the Webcam feed without bottlenecks.
* Recognize images using Inception v3.
* Text to speech with Google TTS API.
* Making a functional model.
* Tuninning the parameters.
* Output visual display of the results.
 
 Dependencies
 ---
This module requires:

* [Python 3.6.1](https://www.python.org/)
* [Tensorflow-gpu 1.0](https://www.tensorflow.org/install/install_windows#requirements_to_run_tensorflow_with_gpu_support)
* [Opencv 3.2](http://opencv.org/)
* [Numpy 1.12](http://www.numpy.org/)
* [Gtts 1.1](https://pypi.python.org/pypi/gTTS)
* [Pygame 1.9](http://www.pygame.org/news)

Usage
---
Just run:
`` python classify_real_time_v2.py``

The output should look like this:


![alt text][image1]

More details
---
For more information, check my medium post [here](https://medium.com/@lucasgago/real-time-image-recognition-and-speech-5545f267f7b3)

Licence
---
This proyect is Copyright © 2016-2017 Lucas Gago. It is free software, and may be redistributed under the terms specified in the [MIT Licence](https://opensource.org/licenses/MIT).
