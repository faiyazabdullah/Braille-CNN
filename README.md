<!DOCTYPE html>
<html lang="en">
<head>
    
</head>
<body>

<h1>BrailleNet: Convolutional Neural Network for Braille Character Classification</h1>

<h2>Introduction</h2>
<p>BrailleNet is a convolutional neural network (CNN) designed for the classification of Braille characters. This repository includes the code for training the model using the Braille Dataset and provides an overview of the network architecture.</p>

<h2>Dataset Preparation</h2>
<ol>
    <li>The Braille Dataset should be organized in the following structure:</li>
    <code>
        dataset/<br>
        └── Braille Dataset/<br>
        &emsp;└── Braille Dataset/<br>
        &emsp;&emsp;├── a1.JPG<br>
        &emsp;&emsp;├── a2.JPG<br>
        &emsp;&emsp;├── ...<br>
        &emsp;&emsp;└── z9.JPG<br>
    </code>
    <li>Create a directory to store images for training and validation:</li>
    <code>$ mkdir ./images/</code>
    <li>Run the script <code>prepare_dataset.py</code> to organize the images into directories corresponding to their respective characters:</li>
    <code>$ python prepare_dataset.py</code>
</ol>

<h2>Training the Model</h2>
<ol>
    <li>Install the required dependencies:</li>
    <code>$ pip install -r requirements.txt</code>
    <li>Run the training script:</li>
    <code>$ python train_model.py</code>
    <li>The trained model will be saved as <code>BrailleNet.h5</code>, and training progress will be stored in the <code>history</code> object.</li>
</ol>

<h2>Model Architecture</h2>
<p>The CNN architecture includes separable convolutional layers, batch normalization, and dropout for robust feature extraction. The model is trained on Braille characters with data augmentation.</p>

<h2>Evaluation</h2>
<ol>
    <li>After training, the model is evaluated on a validation set, and the accuracy is printed.</li>
    <li>To load the trained model and evaluate it:</li>
    <code>$ python evaluate_model.py</code>
</ol>

<h2>Requirements</h2>
<ul>
    <li>Python 3.x</li>
    <li>TensorFlow</li>
    <li>Keras</li>
    <li>Matplotlib (for visualization)</li>
</ul>

<p>Feel free to customize the model architecture or experiment with hyperparameters based on specific requirements.</p>

<h2>License</h2>
<p>This project is licensed under the MIT License - see the <a href="LICENSE">LICENSE</a> file for details.</p>

</body>
</html>
