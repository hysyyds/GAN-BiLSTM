import numpy as np
import pandas as pd
import matplotlib as mlt
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPathCollection
plt.rcParams["font.family"] ="Times New Roman"
plt.rcParams["legend.markerscale"] = 3
plt.rcParams['font.size']=8
def draw():
    size=0.5
    al=0.5
    color1='#9ec57c'
    color2='#0573a8'
    p1=plt.scatter(field['lon'],field['lat'],s=size,color="#449945",label='field',alpha=al)
    p2=plt.scatter(road['lon'],road['lat'],s=size,color="#c22f2f",label='road',alpha=al)
    plt.legend(labels=["field","road"],loc="best",fontsize=10)
    plt.axis('off')


if __name__=="__main__":
    df=pd.read_csv(r"./predict.csv",header=0)
    plt.figure(figsize=(5,5))
    plt.subplot(1,2,1)
    field = df[df['tag'] == 1]
    road = df[df['tag'] == 0]
    dy=-0.1
    plt.title("(a) Ground-truth Trajectory",y=dy)
    draw()
    plt.subplot(1,2,2)
    plt.title("(b) Trajectory Segmented by GAN-BiLSTM",y=dy)
    field = df[df['predict'] == 1]
    road = df[df['predict'] == 0]
    draw()
    plt.savefig("trajectory_new.png",dpi=800, bbox_inches='tight')
    plt.show()
    