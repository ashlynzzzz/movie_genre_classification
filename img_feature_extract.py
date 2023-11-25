import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import os, argparse

def resnet50(remove=True):
    # Load pre-trained ResNet-50 model
    cpt_path = 'model_checkpoint/resnet50-19c8e357.pth'
    cpt_resnet50 = torch.load(cpt_path)
    resnet50 = models.resnet50(weights=None)
    resnet50.load_state_dict(cpt_resnet50)

    if remove:
        # Remove the fully connected layers (classifier)
        resnet50 = torch.nn.Sequential(*list(resnet50.children())[:-1])

    # Set the model to evaluation mode
    resnet50.eval()

    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return resnet50, preprocess

def vgg16(remove=True):
    # Load pre-trained VGG-16 model
    cpt_path = 'model_checkpoint/vgg16-397923af.pth'
    cpt_vgg16 = torch.load(cpt_path)
    vgg16 = models.vgg16(weights=None)
    vgg16.load_state_dict(cpt_vgg16)

    if remove:
        # Remove the fully connected layers (classifier)
        vgg16 = torch.nn.Sequential(*list(vgg16.features.children())[:-1]) 

    # Set the model to evaluation mode
    vgg16.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return vgg16, preprocess

def feature_extract(df, model, preprocess, name, remove=True):
    image_folder = 'movie_images'
    image_files = df['id'].tolist()
    if remove:
        feature_folder = 'image_features_' + name + '_remove'
    else:
        feature_folder = 'image_features_' + name
    os.makedirs(feature_folder, exist_ok=True)

    # Process images in batches
    batch_size = 1000
    features_list = []
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        image_batch = []
        for image_file in batch_files:
            # Load and preprocess the image
            image_path = os.path.join(image_folder, image_file + '.jpg')
            image = Image.open(image_path)
            input_tensor = preprocess(image)
            input = input_tensor.unsqueeze(0)  # Add batch dimension
            image_batch.append(input)

        input_batch = torch.cat(image_batch, dim=0)

        # Extract features using the modified ResNet-50 model
        with torch.no_grad():
            features = model(input_batch)

        # Save features
        features_np = features.squeeze().numpy()
        features_list.append(features_np)
    features_list = np.concatenate(features_list)
    np.save(f'{feature_folder}.npy', features_list)

def main(params):
    df = pd.read_csv('movies.csv')
    if params.model == 'resnet50':
        model, preprocess = resnet50(params.remove)
    elif params.model == 'vgg16':
        model, preprocess = vgg16(params.remove)
    feature_extract(df, model, preprocess, params.model, params.remove)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finetune Image Model")
    parser.add_argument("--model", type=str, default='resnet50')
    parser.add_argument("--remove", type=bool, default=True)
    params, unknown = parser.parse_known_args()
    main(params)