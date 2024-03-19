import json
from nltk_utils import tokenize, stem,bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from model import NeuralNet

with open('./data/intents.json','r') as f:
    intents=json.load(f)


#print(intents)


all_words=[]
tags=[]
#patters and tags
xy=[]
for intent in intents['intents']:
    tag=intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w=tokenize(pattern)
        #w will be a array, so if i append, i will get an
        #array of array, so i will extend
        all_words.extend(w)
        xy.append((w,tag))

#stemming
ignore_words= ['?','!','.',',']
all_words = [stem(w) for w in all_words if w not in ignore_words]

#removing duplicate elements with set, and returning a list with sorted again
all_words=sorted(set(all_words))

tags=sorted(set(tags))
#print(tags)

#bag of words
X_train= []
y_train= []
#we have to unpack the tuple xy
for (pattern_sentence,tag) in xy:
    bag= bag_of_words(pattern_sentence,all_words)
    X_train.append(bag)
    
    label=tags.index(tag) 
    y_train.append(label) #crossentropyloss


X_train=np.array(X_train)
y_train=np.array(y_train)




class ChatDataSet(Dataset):

    def __init__(self):
        self.n_samples=len(X_train)
        self.X_data=X_train
        self.y_data=y_train
        #datasetinindex
    def __getitem__(self,index):
        return self.X_data[index],self.y_data[index]
    
    def __len__(self):
        return self.n_samples
    
#hyperparameteres
batch_size=8
hidden_size=32
output_size=len(tags)
input_size=len(X_train[0])
learning_date=0.001
num_epochs=1000

device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')

model=NeuralNet(input_size,hidden_size,output_size).to(device)


dataset=ChatDataSet()

train_loader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=2)


#loss and optimizer
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=learning_date)

#trainig loop

for epoch in range(num_epochs):
    for(words,labels) in train_loader:
        words= words.to(device)
        labels=labels.to(device)

        #forward
        output=model(words)
        loss=criterion(output,labels)

        #backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch +1) % 100 ==0:
        print(f'epoch {epoch+1}/{num_epochs},loss={loss.item():.4f}')


print(f'final loss ,loss={loss.item():.4f}')

data= {
    'model_state': model.state_dict(),
    'input_size' : input_size,
    'output_size' : output_size,
    'hidden_size' : hidden_size,
    'all_words' : all_words,
    'tags' : tags

}

FILE ='data.pth'
torch.save(data,FILE)

print(f'Training complete, file saved to {FILE}')

