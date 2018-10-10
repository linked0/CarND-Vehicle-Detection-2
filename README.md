# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[hog_image]: ./output_images/hog_car_notcar.png
[hls_image]: ./output_images/HLS_8_8_2_ALL.png
[luv_image]: ./output_images/LUV_8_8_2_ALL.png
[yuv_image]: ./output_images/YUV_8_8_2_ALL.png
[sliding_1]: ./output_images/sliding_1.4_1.png
[sliding_2]: ./output_images/sliding_1.4_2.png
[sliding_3]: ./output_images/sliding_1.4_3.png
[sliding_4]: ./output_images/sliding_1.4_4.png
[sliding_5]: ./output_images/sliding_1.4_5.png
[sliding_6]: ./output_images/sliding_1.4_6.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the second code cell of the IPython notebook

I started by reading in all the `vehicle` and `non-vehicle` images and used 500 sample files each.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][hog_image]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters using scikit-learn SVC classifier and the results are as follows.

|Color Space|Orientation|Pixel per Cell|Cell per Block|HOG Channel|Test Accuracy(0~1.0)|
|:--:|:--:|:--:|:--:|:--:|:--:|
|LUV|8|8|2|0|0.97|
|LUV|8|8|2|1|0.975|
|LUV|8|8|2|2|0.975|
|LUV|8|8|2|ALL|0.97|
|LUV|9|8|2|0|0.965|
|LUV|9|8|2|1|0.945|
|LUV|9|8|2|2|**0.98**|
|LUV|9|8|2|ALL|0.975|
|HLS|8|8|2|0|0.975|
|HLS|8|8|2|1|0.965|
|HLS|8|8|2|2|0.97|
|HLS|8|8|2|ALL|**0.985**|
|HLS|9|8|2|ALL|0.965|
|YUV|8|8|2|ALL|**0.99**|
|YUV|8|8|3|ALL|0.985|
|YUV|9|16|2|ALL|0.975|
|YUV|9|8|2|ALL|0.985|
|YCrCb|8|8|2|ALL|**0.97**|
|HSV|8|8|2|ALL|0.965|
|RGB|8|8|2|ALL|0.955|

<br>
Running the lecture's search_classify.py with the better 3 combinations, the output images are as follows.

|Parameters|Ouput|
|:--:|:--:|
|Color Space:LUV<br>Orientation:8<br>Pixel per Cell:8<br>Cell per Block:2<br> HOG Channel:ALL|![alt text][luv_image]

|Parameters|Ouput|
|:--:|:--:|
|Color Space:YUV<br>Orientation:8<br>Pixel per Cell:8<br>Cell per Block:2<br> HOG Channel:ALL|![alt text][yuv_image]

|Parameters|Ouput|
|:--:|:--:|
|Color Space:HLS<br>Orientation:8<br>Pixel per Cell:8<br>Cell per Block:2<br> HOG Channel:ALL|![alt text][HLS_image]

Final parameters for feature extraction are as follows.<br>
Color Space:LUV<br>Orientation:8<br>Pixel per Cell:8<br>Cell per Block:2<br> HOG Channel:ALL

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

With final parameters found in last section, I started to search best SVM classifier using Scikit-learn GridSearchCV. I started to optimizing the Gamma and C parameters.

When tuning SVM, I can only tune the C parameter with a linear kernal. For a non-linear kernal, I can tune C and gamma.

I started to optimizing the parameters with the following example from Udacity's lecture.
```python
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svr = svm.SVC()
clf = grid_search.GridSearchCV(svr, parameters)
clf.fit(iris.data, iris.target)
```

I compared three classifier from the Scikit-learn, linear SVM, RBF kernal SVM, and LinearSVC. I got the best result as follows.

Best classifier: LinearSVC
Best "C" parameter: 0.1
Best score: 0.936

The code for this step is contained in the fifth, sixth, and seventh code cells of the IPython notebook.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I adapted the method find_cars from Udacity's lecture and changed function arguments to set the final parameters. I didn't change the window size(64x64) of the code from the original method find_cars and explored several scale values. Finally I found that the value 1.4 for the scale value was best for the test images but there were several false positive exampls.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on many scales using LUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][sliding_1]
![alt text][sliding_2]
![alt text][sliding_3]
![alt text][sliding_4]
![alt text][sliding_5]
![alt text][sliding_6]

### se

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

