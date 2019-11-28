import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import os
import multiprocessing as mp

class BranchTraceDataset(torch.utils.data.Dataset):
  """
  Branch trace dataset

  Iterator returns (hash, taken) tuple

  hash = pc ^ bhr
  """
  def __init__(self, trace_path, bhr_len, num_samples):
    super(BranchTraceDataset, self).__init__()
    self.trace_path = trace_path
    self.data = []
    with open(trace_path, 'rb') as trace_file:
      for i, line in enumerate(trace_file):
        # Store data as a list of [<pc>, <is_taken?>, inst_count] lists
        pc, taken, inst_count = (int(x) for x in line.strip().split())
        self.data += [[pc, taken, inst_count]]
      
        if i > num_samples:
          break
    
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

    # Inst count
    inst_count = self.data[idx][2]

    # Build the tensor item
    bpred_idx_tensor = torch.Tensor([[((bpred_idx>>x) & 1) for x in range(self.bhr_len)]])
    label_tensor = torch.LongTensor([label])

    return bpred_idx_tensor, label_tensor, inst_count
  
  def __str__(self):
    return os.path.basename(self.trace_path)

class BPredFPNet(torch.nn.Module):
  def __init__(self, bhr_len):
    super(BPredFPNet, self).__init__()
    self.fc1 = torch.nn.Linear(bhr_len, 2)
    self.lsf = torch.nn.LogSoftmax(1)
    self.lossfunc = nn.CrossEntropyLoss(reduction='sum')
  
  def forward(self, data):
    x = self.fc1(data)
    x = self.lsf(x)
    return x

  def loss(self, prediction, label):
    return self.lossfunc(prediction, label)

def train(model, dataset, optim, trace_file, bhr_len, table_size, num_samples):
  """
  Branch prediction training routine


  Args:

  model -- torch model of predictor
  trace_file -- path to branch trace
  bhr_len -- branch history register len
  table_size -- Number of unique hash values
  """
  try:
    correct = 0
    total = 0
    for idx, (data, label, inst_count) in enumerate(dataset):
      optimizer.zero_grad()
      output = model(data)
      loss = model.loss(output, label)
      loss.backward()
      optimizer.step()

      predicted_idx = torch.argmax(output)

      if label == predicted_idx: correct += 1
      total += 1

      if (idx > num_samples):
        break
  finally:
    print("{}: correct = {} / {}; acc = {:0.2f}%; missPerKI = {:0.3f}".format(
      dataset, correct, total, (correct/total)*100.0, (1000.0 * (total - correct)) / inst_count))

if __name__ == '__main__':
  BHR_LEN = 16
  LR = 0.1
  NUM_SAMPLES = 1000

  btl = BranchTraceDataset(sys.argv[1], BHR_LEN, NUM_SAMPLES)
  model = BPredFPNet(BHR_LEN)
  optimizer = optim.SGD(model.parameters(), lr=LR)
  
  train(model, btl, optimizer, sys.argv[1], BHR_LEN, 1, NUM_SAMPLES)
