### **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./imgs/car_not_car.png
[image2]: ./imgs/HOG_example.png
[image3]: ./imgs/features.png
[image4]: ./imgs/sliding.png
[image5]: ./imgs/bboxes_and_heat.png
[image6]: ./imgs/pipeline_out.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video_output786.mp4

#### [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
##### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.   

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines 25 through 42 of the file called `helper.py`).

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and found Orientation = 9, HOG pixels per cell as 8 and HOG cells per block as 2 to suit my needs.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using in the notebook `VehicleDetection Model Creation.pynb` cell 7. I used GridSearchCV to find the best parameter for Support Vector classifier. Also, the final trained best estimator was stored in a pickle file for later use as shown below.

```
parameters = {'kernel':('linear', 'rbf'), 'C':[0, 10]}
svr = svm.SVC(verbose=1)

print('Starting to do Grid Search for parameters..')
svc = GridSearchCV(svr, parameters)
print('Training the classifier..')
svc.fit(X_train, y_train)

# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
from sklearn.externals import joblib
joblib.dump(svc.best_estimator_, classifer_pickle, compress=9)
print('Model saved to Pickle file')

```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search window positions in lower section of the image. A filter is slided across the selected region with certain overlap percentage. 

![alt text][image4]

I used bounding boxes of size 96x96 size with an overlap percentage of 75%

Each extracted image is resized to 64*64 size. Following which the features are extracted as shown below.

![alt text][image3]


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image5]


---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output786.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

![alt text][image6]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Finding the best parameter was the trickiest part. Also, the training of SVM took a lot of time. I thought its best to use GridSearchCV to find the best parameters. Saving the the best model from the one's found also helped speed up the testing process.

The pipeline might fail in different conditions like rain and snow. 

