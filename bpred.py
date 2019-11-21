import numpy as np
import torch
import sys

class BranchTraceDataset(torch.utils.data.Dataset):
  def __init__(self, trace_path, bhr_len):
    super(BranchTraceDataset, self).__init__()

    self.data = []
    with open(trace_path, 'rb') as trace_file:
      for line in trace_file:
        # Store data as a list of [<pc>, <is_taken?>] lists
        pc, taken = (int(x) for x in line.strip().split())
        self.data += [[pc, taken]]
    
    self.bhr_len = bhr_len
    self.bhr_mask = (1 << bhr_len) - 1

  def __len__(self):
    len(self.data)

  def __getitem__(self, idx):
    # Calculate bhr
    bhr = 0
    for i in range(1, self.bhr_len+1):
      if idx - i < 0:
        taken = torch.tensor(0, dtype=torch.bool)
      else:
        taken = self.data[idx-i][1]

      bhr = (bhr << 1) | int(taken)

    # Prediction index = pc ^ bhr
    bpred_idx = (self.data[idx][0] ^ bhr) & self.bhr_mask

    # Label
    label = self.data[idx][1]

    # Build the tensor item
    bpred_idx_tensor = torch.tensor([((bpred_idx>>x) & 1) for x in range(self.bhr_len)])
    label_tensor = torch.tensor(label)
    return bpred_idx_tensor, label_tensor

if __name__ == '__main__':
  btl = BranchTraceDataset(sys.argv[1], 8)
  for i in range(100):
    print(btl[i])
