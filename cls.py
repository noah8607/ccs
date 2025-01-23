import os
import pickle
import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import AllData
ad = AllData()
types = ad.types

trainingdatax = []
trainingdatay = []

for r in AllData.alldata['train_records']:
    trainingdatax.append(torch.tensor(r['embed'], dtype=torch.float32))
    trainingdatay.append(torch.tensor(r['typeid'], dtype=torch.int64))

model_path = "models/cls/"
device = "cpu"
EBDWIDTH = len(trainingdatax[0])

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.lin1 = nn.Linear(EBDWIDTH, EBDWIDTH)
        # self.bn1 = nn.BatchNorm(EBDWIDTH)
        self.act1 = nn.Softplus()
        self.lin2 = nn.Linear(EBDWIDTH, EBDWIDTH)
        # self.bn2 = nn.BatchNorm(EBDWIDTH)
        self.act2 = nn.Softplus()
        self.lin3 = nn.Linear(EBDWIDTH, EBDWIDTH)
        # self.bn3 = nn.BatchNorm(EBDWIDTH)
        self.act3 = nn.Softplus()
        self.totype = nn.Linear(EBDWIDTH, len(types))

    def forward(self, x):
        x = self.lin1(x)
        # x = self.bn1(x)
        x = self.act1(x)
        x = self.lin2(x)
        # x = self.bn2(x)
        x = self.act2(x)
        x = self.lin3(x)
        # x = self.bn3(x)
        x = self.act3(x)
        x = self.totype(x)
        return x

def loadModel(eval=True):
    global model
    model = Classifier()
    if os.path.exists(model_path+"model.pth"):
        model.load_state_dict(torch.load(model_path+"model.pth", map_location=torch.device('cpu')))
    if eval:
        model.eval()

def saveModel():
    torch.save(model.state_dict(), model_path+"model.pth")

def train():
    if not model:
        Classifier.loadModel()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(250):
        for i in range(len(trainingdatax)):
            optimizer.zero_grad()
            output = model(trainingdatax[i])
            loss = criterion(output, trainingdatay[i])
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} loss: {loss.item()}")

    saveModel()

loadModel(eval=False)

def predict(embedding):
    if not model:
        loadModel(eval=True)
    x = torch.tensor([embedding], dtype=torch.float32)
    x = x.to(device)
    p = model(x)[0]
    psoft = F.softmax(p, dim=0)
    top5p = torch.topk(psoft, 5 if len(psoft)>5 else len(psoft))
    t5 = [types[t] for t in top5p.indices.tolist()]
    p5 = top5p.values.tolist()

    return t5[0], p5[0], t5, p5

if __name__ == "__main__":
    train()