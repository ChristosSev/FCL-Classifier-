import numpy as np
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam,SGD
from torch.autograd import Variable
import torchvision
import pathlib
import torchvision.models as models
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from torchmetrics import ConfusionMatrix

num_epochs=3

#checking for device
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#Transforms
transformer=transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
    transforms.Normalize([0.5,0.5,0.5], # 0-1 to [-1,1] , formula (x-mean)/std
                        [0.5,0.5,0.5])
])


train_path='/home/christos_sevastopoulos/Desktop/KATHARA/Clean_data/Dataset_her/train'
test_path= '/home/christos_sevastopoulos/Desktop/KATHARA/Clean_data/Dataset_uc/val'


train_loader=DataLoader(
    torchvision.datasets.ImageFolder(train_path,transform=transformer),
    batch_size=128, shuffle=True
)
test_loader=DataLoader(
    torchvision.datasets.ImageFolder(test_path,transform=transformer),
    batch_size=64, shuffle=False
)



#calculating the size of training and testing image
train_count=len(glob.glob(train_path+'/**/*.jpg'))
test_count=len(glob.glob(test_path+'/**/*.jpg'))

print(train_count,test_count)


#define the model

model = models.resnet34(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

model.to(device)

optimizer= SGD(model.parameters(),lr=0.001,weight_decay=0.0001)
loss_function=nn.CrossEntropyLoss()


# for name, child in model.named_children():
#     print(name)


#Model training and saving best model

best_accuracy = 0.0

for epoch in range(num_epochs):

    # Evaluation and training on training dataset
    model.train()
    train_accuracy = 0.0
    train_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.cpu().data * images.size(0)
        _, prediction = torch.max(outputs.data, 1)

        train_accuracy += float(torch.sum(prediction == labels.data))

    train_accuracy = train_accuracy / train_count
    train_loss = train_loss / train_count

    # Evaluation on testing dataset
    model.eval()

    test_accuracy = 0.0
    labels_list = []
    prediction_list = []


    for i, (images, labels) in enumerate(test_loader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

            labels_list.extend((labels.cpu()))

        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        prediction_list.extend(prediction.cpu())
        #print(prediction)
        test_accuracy += int(torch.sum(prediction == labels.data))

    test_accuracy = test_accuracy / test_count


    print('Epoch: ' + str(epoch) + ' Train Loss: ' + str(train_loss) + ' Train Accuracy: ' + str(
        train_accuracy) + ' Test Accuracy: ' + str(test_accuracy))


    # Save the best model
    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(), 'best_checkpoint.model')
        best_accuracy = test_accuracy

classes = ('Positive' , 'Negative')

labels_list = np.array(labels_list)
prediction_list = np.array(prediction_list)

#print(labels_list.shape)
#print(prediction_list.shape)

cm = confusion_matrix(labels_list,prediction_list)
print(cm)


# tn, fp, tp, fn= confusion_matrix(labels_list,prediction_list).ravel()
# print(tn,fp,fn,tp)




# import itertools
#
#
# def plot_confusion_matrix(cm,
#                           classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix very prettily.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#
#     # Specify the tick marks and axis text
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=90)
#     plt.yticks(tick_marks, classes)
#
#     # The data formatting
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#
#     # Print the text of the matrix, adjusting text colour for display
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.tight_layout()
#     plt.show()
#
#
#


