# Imports here
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import time
import os.path
import sys
import copy
from PIL import Image
import numpy as np


def main() :
    print("Main")
    args = getArgs()
    processType = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(processType)
    image_datasets, data_loaders = loadData()

    model = initPretrainedModel(args.arch)
    props = {"hiddenUnits" : args.hidden_units, "arch" : args.arch}
    loadClassifier(model, props)

    trainModel(model, args.epochs, processType, data_loaders["train_loader"],data_loaders["valid_loader"],args.print_every)
    testModel(model, processType, data_loaders["test_loader"])
    saveModelToCheckPoint(model, args.save_dir, args.arch, image_datasets["train_data"] )

def getArgs() :
    print("get  Args")
    parser = argparse.ArgumentParser( description="Argument Parser for training model")
    parser.add_argument('data_dir', action="store")
    parser.add_argument("--save_dir" , action="store", default="")
    parser.add_argument("--arch", action="store", default='vgg16' )
    parser.add_argument("--learning_rate", action="store", default=0.003,
                            type=float, help='learning rate (default = 0.003)' )
    parser.add_argument("--hidden_units", action="store", default=512, type=int,
                            help='Pass in an integer (default = 512)')
    parser.add_argument("--epochs", action="store", default=4, type=int, help="Default 4 epochs (int)")
    parser.add_argument("--gpu", action="store_true", help="True is present")
    parser.add_argument("--print_every", action="store", default=30, type=int, help="Steps to print validation (Default 30)")

    pArgs = results = parser.parse_args()

    valid_archs = ["vgg16", "alexnet"]
    if not pArgs.arch in valid_archs :
        print("Invalid arch: %s" % pArgs.arch)
        sys.exit()


    print("Args are all set, beginning training")
    print("Arch = %s" % pArgs.arch)
    print("Learning Rate = %s" % pArgs.learning_rate)
    print("Hidden Units = %i" % pArgs.hidden_units)
    print("Epochs = %s" % pArgs.epochs)
    print("GPU = %s" % pArgs.gpu)
    return pArgs

def loadData() :
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Training data transforms
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    # validation data transforms
    valid_transform =  transforms.Compose([transforms.Resize(255),
                                  transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    # Testing data transforms (same as validation, but why not have two)
    test_transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])


    #Load the datasets with ImageFolder
    image_datasets = {}
    image_datasets['train_data'] = datasets.ImageFolder(train_dir, transform=train_transform)
    image_datasets['valid_data'] = datasets.ImageFolder(valid_dir, transform=valid_transform)
    image_datasets['test_data'] = datasets.ImageFolder(test_dir, transform=test_transform)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    data_loaders = {}
    data_loaders['train_loader'] = torch.utils.data.DataLoader(image_datasets['train_data'], batch_size=50, shuffle=True)
    data_loaders['valid_loader'] = torch.utils.data.DataLoader(image_datasets['valid_data'], batch_size=50)
    data_loaders['test_loader'] = torch.utils.data.DataLoader(image_datasets['test_data'], batch_size=50)
    return image_datasets, data_loaders


#loading model

def initPretrainedModel(archType) :
    if archType == "vgg16" :
        model = models.vgg16(pretrained=True)
    # Not sure how I want to handle other options. Should probably make it so you can choose any model
    # but that might break things downstream
    elif archType == "alexnet" :
        model = models.alexnet()
    else :
        print("Invalid Arch Type [%s] try either (vgg16 or alexnet )" % archType)
        sys.exit()

    print("Arch type = %s " % archType)
    return model


def loadClassifier(model, props) :
    for param in model.parameters():
        param.requires_grad = False

    initInputs = 25088 if props["arch"] == "vgg16"  else 9216
    print("arch %s has %i inputs" %(props["arch"], initInputs))

    classifier = nn.Sequential(nn.Linear(25088,500),
                     nn.ReLU(),
                     nn.Dropout(p=0.05),
                     nn.Linear(500,102),
                     nn.LogSoftmax(dim=1))

    model.classifier = classifier

def trainModel(model, epochs, processType, train_loader, valid_loader, print_every) :
    print("Training model")
    # just hard coding this because it keeps reading cuda as unavailable, not sure why
    # Training plateaus pretty quickly, but seems to get the job done.
    processType = torch.device("cuda")
    print(processType)
    model.to(processType)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
    steps = 0
    running_loss = 0

    for epoch in range(epochs):
        for images, labels in train_loader:
            steps += 1
            device = processType
            images, labels = images.to(device), labels.to(device)

        # training loop

            optimizer.zero_grad()

        #logprob
            logps = model(images)
            # loss from criterion
            loss = criterion(logps, labels)
            # backward pass
            loss.backward()
            optimizer.step()
        # training loss
            running_loss += loss.item()

        #check accuracy on validation data set

            if steps % print_every == 0:
                model.eval()
                valid_loss = 0
                accuracy = 0

                for images, labels in valid_loader:
                    images, labels = images.to(device), labels.to(device)
                #logprob
                    logps = model(images)
                # loss from criterion
                    loss = criterion(logps, labels)
                    valid_loss += loss.item()

                #calculate accuracy

                    ps = torch.exp(logps)

                    top_ps, top_class = ps.topk(1, dim=1)
                    equality = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equality.type(torch.FloatTensor))
                print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(valid_loader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(valid_loader):.3f}")
                running_loss = 0
                model.train()

def testModel(model, processType, test_loader) :
    print("Calculating accuracy")
    # hard coding cuda again
    model.to("cuda")
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

    accuracy = 0
    test_loss = 0
    with torch.no_grad():
        model.eval()
        for images, labels in test_loader:
            images, labels = images.to("cuda"), labels.to("cuda")
            #logprob
            logps = model(images)
            # loss from criterion
            loss = criterion(logps, labels)
            test_loss += loss.item()

            #calculate accuracy

            ps = torch.exp(logps)

            top_ps, top_class = ps.topk(1, dim=1)
            equality = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equality.type(torch.FloatTensor))

    print(f"test accuracy: {accuracy/len(test_loader)*100:.1f}%")



def saveModelToCheckPoint(model, checkPointFile, arch, train_data) :
    print("saving checkpoint")
    model.class_to_idx = train_data.class_to_idx

    # i probably don't need to save all of this, but doing it anyways
    checkpoint = {'model': model,
                  'state_dict': model.state_dict(),
                  'class_to_idx': train_data.class_to_idx}

    torch.save(checkpoint, 'checkpoint.pth')


def load_checkpoint(modelPath):
    print("Loading checkpoint")
    checkpoint = torch.load(modelPath)

    model = models.vgg16(pretrained=True)

    model.class_to_idx = checkpoint['class_to_idx']

    classifier = checkpoint['classifier']

    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])

    return model





# call Main()
if __name__ == '__main__' :
    main()
