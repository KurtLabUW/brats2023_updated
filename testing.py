import torch
from torch import transforms
from torch.utils.data import DataLoader
import datasets


data_dir = ''
batch_size=1

# train_composed = transforms.Compose([trans.CenterCropBySize([128,192,128]), 
#                                               trans.NumpyType((np.float32, np.float32,np.float32, np.float32,np.float32))
#                                               ])
train_set = datasets.BratsDataset(data_dir, mode='train')
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

for imgs, seg, info in train_loader:

    for img in imgs:
        print(img.shape)
    print(seg.shape)
    print(info)