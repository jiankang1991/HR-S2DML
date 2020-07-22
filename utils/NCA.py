import torch
from torch import nn
from torch.autograd import Function
import math

eps = 1e-8

class NCACrossEntropy(nn.Module): 
    ''' \sum_{j=C} log(p_{ij})
        Store all the labels of the dataset.
        Only pass the indexes of the training instances during forward. 
    '''
    def __init__(self, labels, margin=0):
        super(NCACrossEntropy, self).__init__()
        self.register_buffer('labels', torch.LongTensor(labels.size(0)))
        self.labels = labels
        self.margin = margin

    def forward(self, x, indexes):
        # print('x shape: ', x.shape)
        # print('indexes shape: ', indexes.shape)
        
        batchSize = x.size(0)
        n = x.size(1)
        exp = torch.exp(x)
        
        # labels for currect batch
        y = torch.index_select(self.labels, 0, indexes.data).view(batchSize, 1) 
        same = y.repeat(1, n).eq_(self.labels)

        # print('y shape:', y.shape)
        # print('same shape: ', same.shape)

       # self prob exclusion, hack with memory for effeciency
        exp.data.scatter_(1, indexes.data.view(-1,1), 0)

        p = torch.mul(exp, same.float()).sum(dim=1)
        Z = exp.sum(dim=1)

        Z_exclude = Z - p
        p = p.div(math.exp(self.margin))
        Z = Z_exclude + p

        prob = torch.div(p, Z)
        prob_masked = torch.masked_select(prob, prob.ne(0))

        loss = prob_masked.log().sum(0)

        return - loss / batchSize

if __name__ == "__main__":

    from dataGen import DataGeneratorSplitting
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    import numpy as np
    from tqdm import tqdm

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_data_transform = transforms.Compose([
                                            transforms.Resize((256,256)),
                                            transforms.ToTensor(),
                                            normalize])

    train_dataGen = DataGeneratorSplitting(data='/home/jkang/Documents/data/scene', 
                                            dataset='AID', 
                                            imgExt='jpg',
                                            imgTransform=val_data_transform,
                                            phase='train')
    
    trainloader_wo_shuf = DataLoader(train_dataGen, batch_size=64, num_workers=4, shuffle=False, pin_memory=True)

    y_true = []

    for idx, data in enumerate(tqdm(trainloader_wo_shuf, desc="extracting training labels")):

        label_batch = data['label'].to(torch.device("cpu"))
        indexes = data['idx'].to(torch.device("cpu"))

        y_true += list(np.squeeze(label_batch.numpy()).astype(np.float32))
    
    y_true = torch.LongTensor(np.asarray(y_true))

    print(y_true.size())

    criterion = NCACrossEntropy(y_true)

    loss = criterion(torch.randn(64,6993), indexes)






