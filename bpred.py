import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
    label = 1 if (self.data[idx][1] == 1) else -1

    # Build the tensor item
    bpred_idx_tensor = torch.Tensor([((bpred_idx>>x) & 1) for x in range(self.bhr_len)])
    label_tensor = torch.Tensor([label])
    return bpred_idx_tensor, label_tensor

class BPredFPNet(torch.nn.Module):
  def __init__(self, bhr_len):
    super(BPredFPNet, self).__init__()
    self.fc1 = torch.nn.Linear(bhr_len, 1)
    self.lossfunc = nn.MSELoss()
  
  def forward(self, data):
    x = self.fc1(data)
    x = F.sigmoid(x)
    return x

  def loss(self, prediction, label):
    return self.lossfunc(prediction, label)

if __name__ == '__main__':
  btl = BranchTraceDataset(sys.argv[1], 15)
  model = BPredFPNet(15)
  optimizer = optim.SGD(model.parameters(), lr=0.5)

  try:
    correct = 0
    total = 0
    for idx, (data, label) in enumerate(btl):
      optimizer.zero_grad()
      output = model(data)
      loss = model.loss(output, label)
      loss.backward()
      optimizer.step()

      if int(label) == -1:
        correct += 1 if (output < 0) else 0
      else:
        correct += 1 if (output > 0) else 0

      total += 1
  finally:
    print("correct = {} / {}".format(correct, total))
