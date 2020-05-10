# ORIENTME
Estimate the Orientation of a Rubiks Cube


<a href="https://imgflip.com/gif/40vdhh"><img src="https://i.imgur.com/aTlNoBz.gif"></a>

<div>
<h1><b>Introduction</b></h1>
<p>
Solving a rubik’s cube is now too mainstream they say. But what if you have 10,000 pictures of the rubicks cube and you are asked to stitch it all together? This is not a problem we need, this is a problem we deserve!

For input you will be given a large number of images, for about half of them, we have measured the orientation of the cube. But to be able to stitch all those images together, you have to figure out how to predict the orientation of the Rubik's Cube for the rest of the images.
</p>
</div>

<div>
<h1><b>💾 Dataset</b><h1>
<p>
The training dataset consists of 5000 images of size 512x512 with 3 channels each (for RGB). The associated labels is a single continuous variables :

xRot : Orientation of the Cube, in degrees, along an arbitrarily chosen axis (a number between 0 and 360). The axis around which this value is measured, is consitent across the whole of the training and the test set.
The test dataset consists of 5001 images of size 512x512 with 3 channels each (for RGB). The goal of the task is to predict the xRot value of the Rubik's Cube in these test images.
</p>
</div>
<div>
<h1><b>📁 Files</b></h1>
<p>
train.tar.gz - (5000 samples) Tar File containing all the training images, and associated labels
test-images.tar.gz - (5001 samples) Tar file containing all the test images
sample_submission.csv - A sample submission file (with random predictions) to demonstrate the expected file structure by the evaluation setup.
</p>
</div>

