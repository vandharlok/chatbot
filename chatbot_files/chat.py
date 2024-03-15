import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words,tokenize
import torch.nn.functional as F



device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')

with open('./data/intents.json','r') as f:
    intents=json.load(f)

FILE ="./data/data.pth"
data=torch.load(FILE)

input_size = data['input_size']
output_size=data['output_size']
hidden_size=data['hidden_size']
all_words=data['all_words']
tags=data['tags']
model_state=data['model_state']

model=NeuralNet(input_size,hidden_size,output_size).to(device)
model.load_state_dict(model_state)
model.eval()


bot_name= 'Sam'
print('Let s chat!')
while True:
    sentence= input('You: ')
    if sentence == 'quit':
        break

    #tokenize our sentence
    sentence=tokenize(sentence)
    X=bag_of_words(sentence,all_words)
    X=X.reshape(1,X.shape[0])
    X = torch.from_numpy(X).to(device)

    output=model(X)
    _,predicted=torch.max(output,dim=1)
    tag=tags[predicted.item()]

    probs=torch.softmax(output,dim=1)
    prob=probs[0][predicted.item()]

    if prob.item() >0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                print(f'{bot_name}: {random.choice(intent["responses"])}')
    else:
        print(f'{bot_name}: I do not understand...')
    
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    print(f"Predicted tag: {tag}, Confidence: {prob.item()}")

        # Existing code for making a prediction
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # Calculate the softmax probabilities
    softmax_probs = F.softmax(output, dim=1)
    prob = softmax_probs[0][predicted.item()]

    # Print the predicted tag and its confidence
    print(f"Predicted tag: {tag}, Confidence: {prob.item()}")

    # New code to print softmax scores for all tags
    print("Softmax scores for all tags:")
    for idx, (tag, score) in enumerate(zip(tags, softmax_probs[0])):
        print(f"{tag}: {score.item():.4f}")
    