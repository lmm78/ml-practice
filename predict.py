import argparse
import os.path
import sys
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd

def main() :
    print("Calculating predictions")
    args = getArgs()
    print(args)
    if os.path.exists('checkpoint.pth') :
        print('Found your checkpoint')
    else :
        print('Can\'t find your checkpoint. please check the name')
        sys.exit()

    model = loadModel('checkpoint.pth')
    print("Here is your saved model:")
    print("---------------------------")
    print(model)
    model.eval()
    # just hardcoding this because cuda is causing a runtimeError and I don't need the gpu
    model.to('cpu')
    topK = predict(args.image_path, model, args.top_k)
    print_topK(topK)

def predict(image_path, model, topk=5):
    # TODO: Implement the code to predict the class from an image file
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    img = process_image(image_path)

    # Changing from numpy to pytorch tensor
    img = torch.tensor(img)
    img = img.unsqueeze(0)
    img = img.float()


    # Run model to make predictions
    model.eval()
    predictions = model.forward(img)
    predictions = torch.exp(predictions)
    top_preds, top_labs = predictions.topk(5)

    # Detach top predictions into a numpy list
    top_preds = top_preds.detach().numpy().tolist()
    top_labs = top_labs.tolist()

    # Create a pandas dataframe joining class to flower names
    labels = pd.DataFrame({'class':pd.Series(model.class_to_idx),'flower_name':pd.Series(cat_to_name)})
    labels = labels.set_index('class')

    # Limit the dataframe to top labels and add their predictions
    labels = labels.iloc[top_labs[0]]
    labels['predictions'] = top_preds[0]

    return labels

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    #opening image with PIL
    im = Image.open(image)
    # setting transforms
    process = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # processing
    processed_image = process(im)

    return processed_image


def print_topK(topK) :
    print("-------------------------------------------")
    print("Top Predictions")
    print(topK)
    return topK



def initPretrainedModel(archType) :
    if archType == "vgg16" :
        model = models.vgg16(pretrained=True)
    elif archType == "alexnet" :
        model = models.alexnet()
    else :
        print("Invalid Arch Type [%s] try either (vgg16 or alexnet )" % archType)
        sys.exit()

    print("..load arch type %s " %archType)
    return model


def loadModel(checkpointPath) :
    print("Loading checkpoint")
    checkpoint = torch.load(checkpointPath)
    model = checkpoint["model"]
    model.state_dict = checkpoint['state_dict']
    model.class_to_idx = checkpoint['class_to_idx']

    return model


def getArgs() :
    print("Here are the args:")
    parser = argparse.ArgumentParser( description="Argument Parser for Trainer")
    parser.add_argument('image_path', action="store")
    parser.add_argument("check_point" , action="store", default="checkpoint.pth")
    parser.add_argument("--top_k", action="store", default='5' , type=int)
    parser.add_argument("--category_names", action="store", default="cat_to_name.json")
    parser.add_argument("--gpu", action="store_true", help="True is present")
    pArgs = results = parser.parse_args()

    if not os.path.exists(pArgs.image_path) :
        print("This file does not exist : %s" % pArgs.image_path)
        sys.exit()

    #flowers/test/17/image_03906.jpg
    return pArgs


# call Main()
if __name__ == '__main__' :
    main()
