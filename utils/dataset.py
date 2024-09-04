from torch.utils.data import Dataset

class BlockDataset(Dataset):
  def __init__(self, data, block_size) -> None:
    super().__init__()
    self.data = data
    self.block_size = block_size

  def __getitem__(self, idx):
    """Returns a block of text with the specified block size"""
    return self.data[idx:idx+self.block_size], self.data[idx+1:idx+1+self.block_size]

  def __len__(self):
    """Returns the number (-1) of blocks in the dataset"""
    return len(self.data) - self.block_size - 1
