import torch
from trainer import Trainer
import argparse
import opt
from models.lstm import AttBiLSTM
from utils.loader import get_predict_loader
import utils.metrics as metrics
import warnings
import pandas as pd
from tqdm import tqdm
warnings.filterwarnings("ignore")


def train():
    Model = AttBiLSTM(2,opt.emb_size,opt.rnn_size,opt.rnn_layers,opt.dropout)
    trainer = Trainer(Model)
    trainer.start_train()


def predict():
    device = opt.device

    Model = AttBiLSTM(2,opt.emb_size,opt.rnn_size,opt.rnn_layers,opt.dropout).to(device).eval()

    print("Resuming from {}".format(opt.resume_path))
    checkpoint = torch.load(opt.resume_path)
    Model.load_state_dict(checkpoint['state_dict'])

    loader=get_predict_loader(opt.data_dir,step=opt.time_tri,batch_size=opt.batch_size,num_workers=opt.number_workers)

    y_predict=torch.tensor(()).to(device)
    y_true=torch.tensor(()).to(device)
    df=pd.read_csv(opt.data_dir,usecols=["lon","lat","tag"])[["lon","lat","tag"]]
    df["predict"]=0
    tbar = tqdm(loader, desc="\r")
    for batch, (X, y) in (enumerate(loader)):
        with torch.no_grad():
            pred = Model(X)
            y_predict=torch.cat([y_predict,pred.argmax(1)],dim=0)
            y_true=torch.cat([y_true,y],dim=0)
    met=metrics.scores(y_true.cpu().numpy(),y_predict.cpu().numpy())
    print(met)
    df["predict"]=y_predict.cpu().numpy().astype(int)
    df.to_csv("predict.csv",index=False)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'predict'],default='train')
    args = parser.parse_args()
    if args.mode == 'train':
        train()
    else:
        predict()