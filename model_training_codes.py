# custom training function 
from tqdm import tqdm_notebook as tqdm
import numpy as np
import torch
from os import makedirs
from os.path import join

def train_epoch(dataloader,
                epochs:int,
                model,
                optimizer,
                device,
                save_model:bool=False,
                save_model_per_epoch:int=5):
  def one_epoch_train(dataloader,
                      epoch:int,
                      model,
                      optimizer,
                      device):
    total_loss=[]
    total_loss_dict=[]
    one_epoch_bar = tqdm(total=len(dataloader),desc=f"epoch {epoch}",leave=True)
    for batch,(images,targets) in enumerate(dataloader):
      images = [image.to(device) for image in images]
      targets = [{k:torch.tensor(v).to(device) for k, v in t.items()} for t in targets]

      loss_dict = model(images,targets)
      losses = sum([loss for loss in loss_dict.values()])
      losses_dict = [{k:v.item()} for k,v in loss_dict.items()]

      optimizer.zero_grad()
      losses.backward()
      optimizer.step()

      total_loss.append(losses.item())
      total_loss_dict.append(losses_dict)

      one_epoch_bar.update(1)


    return total_loss,total_loss_dict
    
    
  epoch_progressbar = tqdm(total=epochs,desc=f"epochs : ")

  for epoch in range(1,epochs+1):
    epoch_progressbar.update(1)

    total_loss,total_loss_dict = one_epoch_train(dataloader,
                                                  epoch,
                                                  model,
                                                  optimizer,
                                                  device)
    print(f"epoch: {epoch} | loss: {np.mean(total_loss)}")
    if save_model:
        modelsPath = "models"
        makedirs(modelsPath,exist_ok=True)
        if epoch % save_model_per_epoch == 0:
            torch.save(model.state_dict(), join(modelsPath,f"model_{str(epoch)}_loss_{np.mean(total_loss):2f}.torch"))
