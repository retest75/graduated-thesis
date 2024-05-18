import time
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
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

        self.acc = []      # acc
        self.fscore = []   # F-1 score

    def eval_fn(self, epoch):
        """ use fine-tune or linear probe to supervised learning """
        since = time.time()
        total_loss = 0.0
        total_len = 0
        correct = 0

        y_pred = np.array([], dtype=int)
        y_true = np.array([], dtype=int)

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

            # compute y_pred and y_true
            y_pred = np.concatenate([y_pred, pred.cpu().numpy()])
            y_true = np.concatenate([y_true, lbl.cpu().numpy()])

            # print imformation in batch
            print(f"Epoch: [{epoch}] [{total_len:4d}/{self.len}] | Loss: {cur_loss:.6f}")
        
        # learning rate decay
        self.lr.append(self.scheduler.get_last_lr())
        self.scheduler.step()
        
        # compute loss, acc, f-score, and time
        epoch_loss = total_loss / total_len
        epoch_acc = correct.item() / self.len
        epoch_fscore = metrics.f1_score(y_true, y_pred)
        epoch_time = int(time.time() - since)

        self.loss.append(epoch_loss)
        self.acc.append(epoch_acc)
        self.fscore.append(epoch_fscore)
        
        return epoch_loss, epoch_acc, epoch_fscore, epoch_time

    def save_log(self, file: str, model_name: str, phase: str, optim: str, epoch: int, Epoch: int):
        """ save experiment log in epoch
        Parameter
        ------------
        phase: str, only for fine_tune or Linear_probe
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
                f.write(f"Epoch: {epoch+1}/{Epoch} | Loss: {self.loss[epoch]:.6f} | F-1 score: {self.fscore[epoch]:.2f} | Acc: {self.acc[epoch]*100:.2f}%\n")
            else:
                f.write(f"Epoch: {epoch+1}/{Epoch} | Loss: {self.loss[epoch]:.6f} | F-1 score: {self.fscore[epoch]:.2f} | Acc: {self.acc[epoch]*100:.2f}%\n")

    def save_acc(self, file: str):
        """ save entire acc """
        with open(file, mode="a") as f:
            for acc in self.acc:
                f.write(f"{acc}, ")
    
    def save_fscore(self, file: str):
        """ save entire F-1 socre """
        with open(file, mode="a") as f:
            for fscore in self.fscore:
                f.write(f"{fscore}, ")
    

class Testing(Evaluation):
    """ Inherit Evaluation class and use to testing. """
    def __init__(self, device, model, dataset, dataloader, criterion, optimizer=None, scheduler=None):
        super().__init__(device, model, dataset, dataloader, criterion, optimizer, scheduler)
        #---------- inherit these attributes ----------#
        # self.device = device
        # self.model = model
        # self.dataset = dataset
        # self.dataloader = dataloader
        # self.criterion = criterion
        # self.optimizer = optimizer
        # self.scheduler = scheduler

        # self.len = self.dataset.__len__() # length of  testing dataset
        # self.loss = []                    # evaluation loss
        # self.lr = []                      # testing learning rate(備用)
        # self.acc = []                     # testing acc
        #----------------------------------------------#
        
        self.y_prob = np.array([], dtype=np.float32) # predicted probability
        self.y_pred = np.array([], dtype=int)        # predicted label
        self.y_true = np.array([], dtype=int)        # true label
    
    def test_fn(self):
        total_loss = 0.0
        total_len = 0
        corrects = 0

        self.model.eval()
        with torch.no_grad():
            for img, lbl in self.dataloader:
                img, lbl = img.to(self.device), lbl.to(self.device)

                # forward-propagation
                output = self.model(img)

                # compute loss
                loss = self.criterion(output, lbl)
                total_loss += loss.item() * img.size(0)

                # compute corrects and y_pred
                _, pred = torch.max(output, 1)
                corrects += torch.sum(pred == lbl.data)
                pred = pred.cpu().numpy()
                self.y_pred = np.concatenate([self.y_pred, pred])

                # compute y_prob
                prob = torch.softmax(output, dim=1)
                prob = prob[:, 1].cpu().numpy()
                self.y_prob = np.concatenate([self.y_prob, prob])

                # compute y_true
                true = lbl.cpu().numpy()
                self.y_true = np.concatenate([self.y_true, true])

                # print testing information in batch
                total_len += img.size(0)
                cur_loss = total_loss / total_len
                print(f"Batch: [{total_len:4d}/{self.len} | Loss: {cur_loss:.6f}]")

            # compute total loss and acc
            self.loss = total_loss / self.len
            self.acc = corrects.item() / self.len

        return self.loss, self.acc, self.y_pred, self.y_prob
    
    def confusion_matrix(self, pth):
        self.matrix = metrics.confusion_matrix(self.y_true, self.y_pred)
        ax = sns.heatmap(self.matrix, annot=True, xticklabels=['0', '1'], yticklabels=['0', '1'], cmap="OrRd")
        ax.set_title("Confusion matrix")
        ax.set_xlabel("Predict")
        ax.set_ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(pth, "confusion-matrix.png"))
        # plt.show()
        plt.clf()
    
    def compute_fscore(self):
        return metrics.f1_score(self.y_true, self.y_pred)
    
    def compute_pr_curve(self, pth):
        self.precision, self.recall, _ = metrics.precision_recall_curve(self.y_true, self.y_prob)

        # plot P-R curve
        plt.plot(self.recall, self.precision)
        plt.title("P-R curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.savefig(os.path.join(pth, "pr-curve.png"))
        # plt.show()
        plt.clf()
    
    def compute_roc(self, pth):
        self.fpr, self.tpr, _ = metrics.roc_curve(self.y_true, self.y_prob)
        self.auc = metrics.roc_auc_score(self.y_true, self.y_prob)

        # plot ROC curve
        plt.plot(self.fpr, self.tpr, label=f"AUC = {self.auc:.2f}")
        plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Random classifier")
        plt.title("ROC curve")
        plt.legend()
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.savefig(os.path.join(pth, "roc.png"))
        # plt.show()
        plt.clf()
        