# Cat-Dog-Image-Recognition
Deep Leaning: Using keras library, ConVnet and InceptionResnetV2 build a model that can classify cat and dog images.

1. Data: Cat & Dog
The CatDog.mat is a dataset of cats and dogs images. This dataset is structured as a dictionary stored in a Matlab file. It contains four key-value pairs:
• 'X': This key holds a list of numerical features representing image data. The dimensions of this data are (4096, 242), where each column is an image, and each row is a vector containing information about that image.
• 'G': This key is associated with labels, containing a list of binary data. The value '0' represents a 'cat', while '1' represents a 'dog'. This key provides the true labels for each corresponding image in the dataset.
• 'nx' and 'ny': These keys store integer values that denote the width ('nx') and height ('ny') of each image in the dataset. These values represent the dimensions of the images, helping to understand the size of each image in terms of its width and height.

2. Data Preprocessing:
2.1 Data Loading: Load the cat and dog dataset using loadmat from the mat4py library.
2.2 Data Preparation: Preprocess the images, resize them to a uniform size, normalize pixel values, and split the dataset into training, validation, and test sets. Here are steps to preprocess the image data before feeding the data to our model:
• Transpose Matrix 'X': This step involves rearranging the matrix 'X' from a dimension of (4096, 242) to (242, 4096). Transposing the matrix helps in organizing the data where each row now represents the vector of information for each image, totaling 242 rows.
• Normalizing 'X': Normalization is performed to bring all pixel values within a similar scale (e.g., [0, 1] or [-1, 1]). This process stabilizes computations and prevents numerical instability issues that might arise from computations involving large values. Achieving normalization involves scaling all values to fit within the range of [0, 1] using a method like minimum and maximum normalization.
• Reshaping 'X': As the image data is currently in an array shape, reshaping 'X' is necessary to transform it into images. Adjusting the dimensions is crucial to align the data with the expected input shape of the model. For instance, if the InceptionResNetV2 architecture expects images sized (75, 75, 1), resizing the images accordingly becomes important.
• Convert to RGB Scale: Converting images from grayscale to RGB scale might be necessary for models pre-trained on RGB images, such as InceptionResNetV2. However, if color information isn't pivotal for your classification task, using grayscale images might suffice.
• Transpose Image: Rotating images by 90 degrees clockwise, achieved by swapping the width and height dimensions, aids in visualizing the images more conveniently.
• Split data into train and test set: with the train size is 80% of the data, and test size is 20% of the data.

3. Build a model for predicton:
   Model Selection: As mentioned above, Inception-ResNet-v2 is a convolutional neural network that is trained on more than a million images from the ImageNet database developed by Google.
   In this project, I import the InceptionResNetV2 architecture from Keras applications and use it as the based-model for transfer learning to classify Cat and dog images.
   Fine-tuning: Freeze certain layers of the pre-trained InceptionResNetV2 model to retain learned features and add custom fully connected layers for classification. This technique is o􀅌en used when using pre-trained models to avoid overfi􀆫ng and retain the learned features.
• Input Preprocessing: Normalizes the input data using batch normalization and preprocesses it for compatibility with the pre-trained InceptionResNetV2 model.
• Feature Extraction: Passes the preprocessed input through the base model to extract higher-level features. This part utilizes the convolutional layers of the InceptionResNetV2 model.
• Flatening: Flatens the extracted features into a vector to prepare them for fully connected layers.
• Activation Functions: activation functions are used to transform an input signal into an output signal. Typically, ReLU (Rectified Linear Activation) is used in intermediate layers, while Softmax is employed in the output layer for classification tasks. Sigmoid activation function: Because it is a non-linear functon, it is the most often utilized activation function. The sigmoid function changes data in the 0 to 1 range and it is widely used for binary classification. In this model, in the fully connected layer (dense layer), we use ReLU activation (Rectified Linear Activation) with 512 neurons. This layer aims to learn complex paterns from the extracted features. And in the output layer, we use a sigmoid activation function with a single output (n_out = 1) for binary classification. This layer produces a single output representing the probability of belonging to the positive class.
