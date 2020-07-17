import os
import yaml
import argparse
import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter

from utils import get_train_valid_loader, get_test_loader

class ConvNet(nn.Module):

    def __init__(self, num_classes):
        super(ConvNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(8, 8), stride=(4, 4)),  # output: (16, 20, 20)
            nn.Conv2d(16, 32, kernel_size=(4, 4), stride=(2, 2)), # output: (32, 9, 9)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, image):
        return self.classifier(self.encoder(image))

    
class Trainer:
    def __init__(self, config):

        self.device = config["device"]

        self.model = ConvNet(num_classes=config["num-classes"])
        self.model.to(self.device)

        if config["resume"]:
            print("> Loading Checkpoint")
            self.model.load_state_dict(T.load(config["load-path"]))

        self.train_loader, self.val_loader = get_train_valid_loader(
            config["data-path"], 
            config["num-classes"],
            config["batch-size"], 
            config["val-batch-size"], 
            config["augment"],
            config["seed"], 
            config["valid-size"], 
            config["shuffle"],
            config["num-workers"]
        )

        self.test_loader = get_test_loader(
            config["data-path"], 
            config["num-classes"],
            config["batch-size"], 
            config["shuffle"],
            config["num-workers"], 
            config["pin-memory"]
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optim = T.optim.AdamW(self.model.parameters(), lr=config["lr-init"], weight_decay=config["weight-decay"])
    
        self.writer = SummaryWriter(log_dir=os.path.join("logs", config["run-title"]))
        self.reduce_lr = T.optim.lr_scheduler.ReduceLROnPlateau(self.optim, factor=config["lr-factor"], patience=config["lr-patience"], min_lr=config["lr-min"])

        self.stopping_patience = config["stopping-patience"]
        self.stopping_delta = config["stopping-delta"]

        self.filepath = os.path.join(config["save-path"], config["run-title"], config["run-title"]+".pt")

    def train_epoch(self, epoch, pbar):
        self.model.train()
        train_loss = np.zeros(len(self.train_loader))
        for i, (inputs, labels) in enumerate(self.train_loader):

            inputs = inputs.float().to(self.device)
            labels = labels.to(self.device)

            self.optim.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optim.step()

            train_loss[i] = loss.item() 
            pbar.set_description(f"Epoch {epoch} | Loss: {train_loss[:i].sum()/(i+1):.4f} | ({i}/{len(self.train_loader)})")

        val_loss = self.validate_epoch()
        return train_loss.mean(), val_loss
 
    def validate_epoch(self):
        self.model.eval()
        val_loss = np.zeros(len(self.val_loader))
        for i, (inputs, labels) in enumerate(self.val_loader):

            inputs = inputs.float().to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            val_loss[i] = loss.item()

        return val_loss.mean()

    def evaluate(self, load_path):
        total, correct = 0, 0
        self.model.load_state_dict(T.load(load_path, map_location=T.device(self.device)))
        self.model.eval()
        for data in self.test_loader:
            images, labels = data[0].to(self.device), data[1].to(self.device)
            outputs = self.model(images)
            _, predicted = T.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return correct / total

    def train(self, epochs):
        stopping_counter = 0
        best_val_loss = np.inf 
        progress = tqdm(range(epochs))
        for epoch in progress: 

            ########## Training ##########            
            train_loss, val_loss = self.train_epoch(epoch, progress)

            self.writer.add_scalar("loss/train", train_loss, epoch)
            self.writer.add_scalar("loss/val", val_loss, epoch)

            progress.write(f"Epoch {epoch}/{epochs}\t| Train Loss {train_loss:.5f} | Val Loss {val_loss:.5f}")

            if val_loss < best_val_loss and abs(val_loss-best_val_loss) > self.stopping_delta :
                stopping_counter = 0
                best_val_loss = val_loss
                T.save(self.model.state_dict(), self.filepath)
            else:
                stopping_counter += 1

            if stopping_counter > self.stopping_patience:
                break

        self.writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Paramaters')
    parser.add_argument('-c', '--config',  type=str, default="config.yaml", help='path of config file')
    args = parser.parse_args()

    with open(args.config, 'r', encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    exp_path = os.path.join(config["save-path"], config["run-title"])
    if not os.path.isdir(exp_path): os.mkdir(exp_path)

    trainer = Trainer(config)

    if config["train"]:
        print("> Training")
        trainer.train(config["epochs"])

    if config["test"]:
        print("> Testing")
        acc = trainer.evaluate(config["load-path"])
        print(f"Testing Accuracy: {acc*100:.4f}%") 
