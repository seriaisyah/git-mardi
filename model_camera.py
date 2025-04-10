import torch
import torchvision.transforms as transforms
from torchvision import models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet50
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
import cv2

# Define the batch size
batch_size = 32

# Define the data transformations - flip, etc
train_transforms = transforms.Compose([
    transforms.RandomAffine(degrees=0, shear=10),  # shear_range
    transforms.RandomHorizontalFlip(),  # horizontal_flip
    transforms.Resize((224, 224)),  # target_size
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Simplified normalization
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # target_size
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Simplified normalization
])

import torch
import torch.nn as nn
import torchvision.models as models

class CustomResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomResNet50, self).__init__()
        # Load pre-trained ResNet50 model without the top (fully connected) layer
        self.conv_base = models.resnet50(pretrained=True)
        for param in self.conv_base.parameters():
            param.requires_grad = False

        # Add global average pooling
        self.conv_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Replace the fully connected layer
        num_ftrs = self.conv_base.fc.in_features
        self.conv_base.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes) # define how many classes
        )

    def forward(self, x):
        if len(x.shape) == 5:  # If the input has shape [batch_size, 32, 3, 224, 224]
            x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])  # Flatten the batch and sequence dimensions
        x = self.conv_base(x)
        return x

# Instantiate the model
model = CustomResNet50()

# Move model to GPU if available - force data to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model=torch.load('testing_images.pth')
model.eval()  # Set the model to evaluation mode

validation_img_paths = ['My_data/botol1jpeg','My_data/pen1.png', 'My_data/botol2.jpeg']

# Load and transform images
img_list = [val_transforms(Image.open(img_path).convert("RGB")) for img_path in validation_img_paths]

# Check if images are transformed correctly
for img in img_list:
    print(type(img))  # Should print <class 'torch.Tensor'>

#Define a function to get the predicted class
def get_prediction_class(prediction_tensor):
    if prediction_tensor[0][1] > prediction_tensor[0][0]: # largest alphabet [0][1]
        return "pen"
    else:
        return "botol"
    

import torch
from PIL import Image

# Evaluate the model on each image
model.eval()

# Determine the device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to the selected device
model = model.to(device)

for image_path in validation_img_paths:
    img = Image.open(image_path).convert("RGB")
    
    # Transform the image and add batch dimension
    img_tensor = val_transforms(img).unsqueeze(0)
    
    # Move the tensor to the same device as the model
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        output = model(img_tensor) #convert image dalam bentuk tensor

    # Convert the output tensor into human-readable class label
    predicted_class = get_prediction_class(output)

    # Print the prediction result
    print(f"Image: {image_path}, Predicted class: {predicted_class}")

    import matplotlib.pyplot as plt
import torch
from PIL import Image

# Determine the device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to the selected device
model = model.to(device)
model.eval()

# Define the figure and axes
fig, axs = plt.subplots(1, len(validation_img_paths), figsize=(20, 5))

# Iterate over each image, make predictions, and plot
for i, image_path in enumerate(validation_img_paths):
    # Load and preprocess the image
    img = Image.open(image_path).convert("RGB")
    img_tensor = val_transforms(img).unsqueeze(0)  # Add batch dimension
    
    # Move the input tensor to the same device as the model
    img_tensor = img_tensor.to(device)

    # Evaluate the model on the image
    with torch.no_grad():
        output = model(img_tensor)

    # Convert the output tensor into human-readable class label
    predicted_class = get_prediction_class(output)

    # Plot the image
    axs[i].imshow(img)
    axs[i].axis('off')
    axs[i].set_title(f"Predicted: {predicted_class}")

# Show the plot
plt.show()

import cv2

def main():
    # Simplified GStreamer pipeline for IMX219 camera
    gst_pipeline = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=640, height=480, format=NV12 ! "
        "nvvidconv ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "appsink"
    )

    # Create a VideoCapture object with the GStreamer pipeline
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Failed to open camera.")
        print("GStreamer pipeline:", gst_pipeline)
        return

    # Create an OpenCV window
    cv2.namedWindow("Camera Stream", cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            # Capture a frame from the camera
            ret, frame = cap.read()

            if not ret:
                print("Failed to grab frame.")
                break

            # Display the frame in the OpenCV window
            cv2.imshow("Camera Stream", frame)

            # Check for exit key (ESC)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Failed to open camera.")
        return

    cv2.namedWindow("Camera Stream", cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Process frame through model
            predictions = process_frame(frame)
            print("Model output:", predictions)  # Modify based on your model's output format

            # Display the frame
            cv2.imshow("Camera Stream", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()