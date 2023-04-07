import torch
import torch.nn as nn
from pprint import pprint
import torch.optim as optim
from tqdm.auto import tqdm
from collections import defaultdict
from models import VecotrQuantizerAE

from utils import VQVAE_cfg,get_data_loaders
from utils import reconstruct , show_2_batches
from utils import plot_history_train_val

# load the config
cfg = VQVAE_cfg()
print("Params:")
pprint(cfg,indent=3)



# load the dataloders
train_loder, val_loder = get_data_loaders()

# load model
model = VecotrQuantizerAE(
    num_downsamplings=cfg.MODEL.NUM_DOWNSAMPLINGS,
    latent_channels=cfg.MODEL.LATENT_CHANNELS,
    num_embeddings=cfg.MODEL.NUM_EMBEDDINGS,
    channels=cfg.MODEL.ENCODER_CHANNELS,
    in_channels=cfg.DATASET.IMAGE_CHANNELS,
)
model = model.to(cfg.TRAIN.DEVICE)

print("Number of parameters: {:,}".format(sum(p.numel() for p in model.parameters())))


class VAELoss(nn.Module):
    def __init__(self, λ=1.0):
        super().__init__()
        self.λ = λ
        self.reconstruction_loss = nn.MSELoss()

    def forward(self, outputs, target):
        output, vq_loss = outputs
        reconst_loss = self.reconstruction_loss(output, target)

        loss = reconst_loss + self.λ * vq_loss
        return {"loss": loss, "reconstruction loss": reconst_loss, "VQ loss": vq_loss}

class AverageLoss:
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.num_samples = 0
        self.total_loss = 0.0

    def update(self, data):
        batch_size = data["batch_size"]
        self.num_samples += batch_size
        self.total_loss += batch_size * data[self.name]

    def compute(self):
        avg_loss = self.total_loss / self.num_samples
        metrics = {self.name: avg_loss}
        return metrics


class Manager:
    def __init__(self,model, loss, optimizer, train_loder, val_loder, device,        batch_scheduler=None,
        epoch_scheduler=None) -> None:
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loder
        self.val_loader = val_loder
        self.device = device
        
        self.batch_scheduler = batch_scheduler
        self.epoch_scheduler = epoch_scheduler
        # history and metrics
        self.history = defaultdict(list)
        self.metrics = [
            AverageLoss(x) for x in ["loss", "reconstruction loss", "VQ loss"]
        ]
        
    def log_metrics(self, metrics, name):
        print(f"{name}: ", end='', flush=True)
        for key, val in metrics.items():
            self.history[name + ' ' + key].append(val)
            print(f"{key} {val:.3f} ", end='')
               
    def fit(self,epochs):
        ######################### Training #####################
        
        for epoch in range(1, epochs + 1):
            print(f"Epoch : {epoch}/{epochs}")
        
            self.model.train()
            for metric in self.metrics:
                metric.reset()
            train_bar = tqdm(self.train_loader, desc=f"Training:")   
                     
            for batch_idx , batch in enumerate(train_bar):
                images = batch.to(self.device)
                outputs = self.model(images)
                losses = self.loss(outputs, images)
                
                
                self.optimizer.zero_grad()
                losses["loss"].backward()
                self.optimizer.step()

                if self.batch_scheduler is not None:
                    self.batch_scheduler.step()
                    
                data = {k: v.item() for k, v in losses.items()}
                data["batch_size"] = len(images)

                for metric in self.metrics:
                    metric.update(data)
                    
                train_bar.set_postfix(
                    loss = losses['loss'].item(), 
                    reconstruction_loss =losses['reconstruction loss'].item(),
                    VQ_loss = losses['VQ loss'].item()
                )



            summary = {}
            for metric in self.metrics:
                summary.update(metric.compute())
            self.log_metrics(summary, "train")
            
            
            ########################### TEST ##########################
            
            self.model.eval()
            for metric in self.metrics:
                metric.reset()
            test_bar = tqdm(self.val_loader, desc=f"Testing:")
            
            with torch.no_grad():
                
                for batch_idx , batch in enumerate(test_bar):
                    images = batch.to(self.device)
                    outputs = self.model(images)
                    losses = self.loss(outputs, images)
                    
                    data = {k: v.item() for k, v in losses.items()}
                    data["batch_size"] = len(images)

                    for metric in self.metrics:
                        metric.update(data)
                        
                    train_bar.set_postfix(
                        loss = losses['loss'].item(), 
                        reconstruction_loss =losses['reconstruction loss'].item(),
                        VQ_loss = losses['VQ loss'].item()
                    )

                        
                summary = {}
                for metric in self.metrics:
                    summary.update(metric.compute())
                self.log_metrics(summary, "val")

        
                # plotting each epoch:
                test_batch = next(iter(val_loder))
                reconstructed_batch = reconstruct(model,test_batch,device = cfg.TRAIN.DEVICE)
                show_2_batches(test_batch[:64], reconstructed_batch[:64], f"Validation Images_epoch_{str(epoch)}", f"Reconstructed Images__{str(epoch)}",epoch)
                
                torch.save(model.state_dict(), str('result/final_model.pt'))




loss = VAELoss(λ=0.1)
optimizer = optim.AdamW(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE, weight_decay=cfg.TRAIN.WEIGHT_DECAY)

lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.TRAIN.LEARNING_RATE,
                                             steps_per_epoch=len(train_loder), epochs=cfg.TRAIN.EPOCHS)
learner = Manager(model, loss, optimizer, train_loder, val_loder, cfg.TRAIN.DEVICE, batch_scheduler=lr_scheduler)

learner.fit(cfg.TRAIN.EPOCHS)


plot_history_train_val(learner.history, 'loss')
plot_history_train_val(learner.history, 'reconstruction loss')
plot_history_train_val(learner.history, 'VQ loss')