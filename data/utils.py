import torch
from torch.utils.data.dataset import Dataset


def one_batch_dataset(dataset, batch_size):
    print("==> Grabbing a single batch")

    perm = torch.randperm(len(dataset))
    # 等价于产生一波随机数
    '''np.shuffle(np.arange(0,len(dataset))'''
    # 得到1个batch_size的数据
    one_batch = [dataset[idx.item()] for idx in perm[:batch_size]]

    class _OneBatchWrapper(Dataset):
        def __init__(self):
            self.batch = one_batch
        # __getitem__就是索引的函数 能得到索引为index的数据
        def __getitem__(self, index):
            return self.batch[index]

        def __len__(self):
            return len(self.batch)

    return _OneBatchWrapper()
