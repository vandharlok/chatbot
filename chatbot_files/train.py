"""
import json
from nltk_utils import tokenize, stem,bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from model import NeuralNet
from sklearn.metrics import accuracy_score

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
    def __init__(self, X_data, y_data):
        self.X_data = torch.FloatTensor(X_data)  
        self.y_data = torch.LongTensor(y_data)

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
    
    def __len__(self):
        return len(self.X_data)
    
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

# Inicialização
train_losses = []
valid_losses = []
valid_accuracy = []

#trainig loop
#o loop certo aki

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


#acaba aki
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for words, labels in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(words)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

    # Cálculo da perda média de treinamento
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # "Validação" - usando um subconjunto aleatório dos dados de treinamento
    model.eval()
    valid_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for words, labels in train_loader:  # Idealmente, use um verdadeiro conjunto de validação
            words = words.to(device)
            labels = labels.to(device)
            output = model(words)
            loss = criterion(output, labels)
            valid_loss += loss.item()
            
            # Calculando a acurácia
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    valid_loss /= len(train_loader)
    valid_losses.append(valid_loss)
    valid_accuracy.append(correct / total)

    if (epoch +1) % 100 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_accuracy[-1]:.4f}')

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

"""
import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from model import NeuralNet

# Carregando os dados
with open('./data/intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

# Preparação dos dados
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Dividindo os dados em treinamento e validação
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Definição do dataset
class ChatDataSet(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = torch.FloatTensor(X_data)
        self.y_data = torch.LongTensor(y_data)

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
    
    def __len__(self):
        return len(self.X_data)
    
# Hiperparâmetros
batch_size = 8
hidden_size = 32
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = 0.001
num_epochs = 1000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Criando datasets e DataLoaders para treinamento e validação
train_dataset = ChatDataSet(X_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

val_dataset = ChatDataSet(X_val, y_val)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Definição de loss e optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Loop de treinamento e validação
train_losses = []
val_losses = []
val_accuracy = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for words, labels in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(words)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    model.eval()
    valid_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for words, labels in val_loader:
            words = words.to(device)
            labels = labels.to(device)
            output = model(words)
            loss = criterion(output, labels)
            valid_loss += loss.item()
            
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    valid_loss /= len(val_loader)
    val_losses.append(valid_loss)
    val_accuracy.append(correct / total)

    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {val_accuracy[-1]:.4f}')

# Salvando o modelo treinado
data = {
    'model_state': model.state_dict(),
    'input_size': input_size,
    'output_size': output_size,
    'hidden_size': hidden_size,
    'all_words': all_words,
    'tags': tags
}

FILE = 'data.pth'
torch.save(data, FILE)

print(f'Training complete, file saved to {FILE}')