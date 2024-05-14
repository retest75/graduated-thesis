import time
import os
import torch

class Training():
    def __init__(self, device, model, dataset, dataloader, criterion, optimizer, scheduler):
        self.device = device
        self.model = model
        self.dataset = dataset
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        # attributes which are used to train_fn
        self.len = self.dataset.__len__() # length of dataset
        self.loss = []                    # save each epoch loss
        self.lr = []                      # save epoch learning rate and use to check(備用)
    
    def train_fn(self, epoch):
        """ used to self supervised learning """
        since = time.time()
        total_loss = 0.0
        total_len = 0

        self.model.train()
        for img, _ in self.dataloader:
            img[0] = img[0].to(self.device)
            img[1] = img[1].to(self.device)

            # forward-propagation
            p1, p2, z1, z2 = self.model(img[0], img[1])

            # compute loss
            loss = -(self.criterion(p1, z2).mean() + self.criterion(p2, z1).mean()) * 0.5

            # backward-propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # computer current average loss in each batch
            total_loss += loss.item() * img[0].size(0)
            total_len += img[0].size(0)
            cur_loss = total_loss / total_len
            
            # print information in batch
            print(f"Epoch: [{epoch}] [{total_len:4d}/{self.len}] | Loss: {cur_loss:.6f}")

        # learning rate decay
        self.lr.append(self.scheduler.get_last_lr())
        self.scheduler.step()

        # compute loss and time
        epoch_time = int(time.time() - since)
        epoch_loss = total_loss / total_len
        self.loss.append(epoch_loss)
        
        return epoch_loss, epoch_time
    
    def save_checkpoint(self, path: str, phase: str, epoch: int, best_param=None):
        """ save checkpoint or best parameter """
        if best_param:
            dst = os.path.join(path, f"{phase}-Epoch[{(epoch+1):02d}]-Loss[{self.loss[epoch]:.6f}](Best).pt")
            torch.save(best_param, dst)
        else:
            dst = os.path.join(path, f"{phase}-Epoch[{(epoch+1):02d}]-Loss[{self.loss[epoch]:.6f}].pt")
            param = self.model.state_dict()
            torch.save(param, dst)
    
    def get_param(self):
        """ get hyper-parameter of optimizer """
        groups = self.optimizer.param_groups[0]
        
        return groups["initial_lr"], groups["momentum"], groups["weight_decay"]

    def save_log(self, file: str, model_name: str, phase: str, optim: str, epoch: int, Epoch: int):
        """ save experiment log in epoch
        Parameter
        ------------
        epoch : int, current epoch
        Epoch : int, entire training epoch
        """
        lr, momentum, weight_decay = self.get_param()
        with open(file, mode="a") as f:
            if (epoch+1) == 1:
                f.write("======== Basic configuration ========\n")
                f.write(f"Model: {model_name}\n")
                f.write(f"Phase: {phase}\n")
                f.write(f"Optimizer: {optim}\n")
                f.write(f"Learing rate = {lr}, Momentun = {momentum}, Weight decay = {weight_decay}\n")
                f.write("=====================================\n")
                f.write(f"Epoch: {epoch+1}/{Epoch} | Loss: {self.loss[epoch]:.6f}\n")
            else:
                f.write(f"Epoch: {epoch+1}/{Epoch} | Loss: {self.loss[epoch]:.6f}\n")
    
    def save_loss(self, file: str):
        """ save entire loss """
        with open(file, mode="a") as f:
            for loss in self.loss:
                f.write(f"{loss}, ")

class Evaluation(Training):
    def __init__(self, device, model, dataset, dataloader, criterion, optimizer, scheduler=None):
        """ 
        Inherit Training class and use to evaluation by fine-tune or linear probe.
        If you want to switch phase between fine-tine and linear probe, don't forget initializing
        model, optimizer, sheduler, loss, acc
        """
        super().__init__(device, model, dataset, dataloader, criterion, optimizer, scheduler)
        #---------- inherit these attributes ----------#
        
        # self.device = device
        # self.model = model
        # self.dataset = dataset
        # self.dataloader = dataloader
        # self.criterion = criterion
        # self.optimizer = optimizer
        # self.scheduler = scheduler

        # self.len = self.dataset.__len__() # length of  evaluation dataset
        # self.loss = []                    # evaluation loss
        # self.lr = []                      # evaluation learning rate(備用)
        #----------------------------------------------#

        self.acc = []                       # acc

    def eval_fn(self, epoch):
        """ use fine-tune or linear probe to supervised learning """
        since = time.time()
        total_loss = 0.0
        total_len = 0
        correct = 0

        self.model.train()
        for (img, lbl) in self.dataloader:
            img = img.to(self.device)
            lbl = lbl.to(self.device)

            # forward-propagation
            output = self.model(img) # output size: [batch, classes], by default classes = 2

            # compute loss
            loss = self.criterion(output, lbl)

            # backward-propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # compute prediction and correct
            _, pred = torch.max(output, 1)
            correct += torch.sum(pred == lbl.data)

            # compute average loss in each batch
            total_loss += loss.item() * img.size(0)
            total_len += img.size(0)
            cur_loss = total_loss / total_len

            # print imformation in batch
            print(f"Epoch: [{epoch}] [{total_len:4d}/{self.len} | Loss: {cur_loss:.6f}]")
        
        # learning rate decay
        self.lr.append(self.scheduler.get_last_lr())
        self.scheduler.step()
        
        # compute loss, acc and time
        epoch_loss = total_loss / total_len
        epoch_acc = correct / self.len
        epoch_time = int(time.time() - since)

        self.loss.append(epoch_loss)
        self.acc.append(epoch_acc)
        
        return epoch_loss, epoch_acc, epoch_time

    def save_log(self, file: str, model_name: str, phase: str, optim: str, epoch: int, Epoch: int):
        """ save experiment log in epoch
        Parameter
        ------------
        phase: str, only for fine_tune or Linear_probe
        epoch : int, current epoch
        Epoch : int, entire training epoch
        """
        lr, momentum, weight_decay = self.get_param()
        with open(file, mode="a") as f:
            if (epoch+1) == 1:
                f.write("======== Basic configuration ========\n")
                f.write(f"Model: {model_name}\n")
                f.write(f"Phase: {phase}\n")
                f.write(f"Optimizer: {optim}\n")
                f.write(f"Learing rate = {lr}, Momentun = {momentum}, Weight decay = {weight_decay}\n")
                f.write("=====================================\n")
                f.write(f"Epoch: {epoch+1}/{Epoch} | Loss: {self.loss[epoch]:.6f} | Acc: {self.acc[epoch]*100:.2f}%\n")
            else:
                f.write(f"Epoch: {epoch+1}/{Epoch} | Loss: {self.loss[epoch]:.6f} | Acc: {self.acc[epoch]*100:.2f}%\n")

    def save_acc(self, file: str):
        """ save entire acc """
        with open(file, mode="a") as f:
            for acc in self.acc:
                f.write(f"{acc}, ")