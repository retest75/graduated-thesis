import time
import os
import torch

class Training():
    def __init__(self, device, model, dataset, dataloader, criterion, optimizer, scheduler=None):
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

            # backward-prppagation
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
    
    def save_log(self, file: str, model_name: str, phase: str, optim: str, epoch: int, Epoch: int):
        """ save experiment log in epoch
        Parameter
        ------------
        epoch : int, current epoch
        Epoch : int, entire training epoch
        """
        with open(file, mode="a") as f:
            if (epoch+1) == 1:
                f.write("======== Basic configuration ========\n")
                f.write(f"Model: {model_name}\n")
                f.write(f"Phase: {phase}\n")
                f.write(f"Optimizer: {optim}\n")
                f.write("=====================================\n")
                f.write(f"Epoch: {epoch+1}/{Epoch} | Loss: {self.loss[epoch]:.6f}\n")
            else:
                f.write(f"Epoch: {epoch+1}/{Epoch} | Loss: {self.loss[epoch]:.6f}\n")
    
    def save_loss(self, file: str):
        """ save entire loss """
        with open(file, mode="a") as f:
            for loss in self.loss:
                f.write(f"{loss}, ")