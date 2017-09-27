from dataset import CamVid
from enet import Enet

# initialize dataset and net
camvid = CamVid()
net = Enet(camvid)

# Start net training
net.train()