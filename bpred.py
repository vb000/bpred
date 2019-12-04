import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import os
import multiprocessing as mp
import matplotlib.pyplot as plt

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

    # PC
    pc = self.data[idx][0]

    # Label
    label = self.data[idx][1]

    # Inst count
    inst_count = self.data[idx][2]

    # Build the tensor item
    bhr_tensor = torch.Tensor([[((bhr>>x) & 1) for x in range(self.bhr_len)]])
    label_tensor = torch.LongTensor([label])

    return pc, bhr_tensor, label_tensor, inst_count
  
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

def train(pid, trace_file, lr, bhr_len, table_size, num_samples, results):
  """
  Branch prediction training routine


  Args:

  pid -- process id
  lr -- learning rate
  trace_file -- branch trace data file
  bhr_len -- branch history register len
  table_size -- Number of unique hash values
  num_samples -- Number of samples to train on
  results -- result dict
  """
  dataset = BranchTraceDataset(trace_file, bhr_len, num_samples)

  mdl_table = []
  optim_table = []
  for i in range(table_size):
    model = BPredFPNet(BHR_LEN)
    mdl_table += [model]
    optim_table += [optim.SGD(model.parameters(), lr=lr)]
  
  try:
    correct = 0
    total = 0
    mpkis = []
    prev_correct = 0
    prev_inst_count = 0
    for idx, (pc, data, label, inst_count) in enumerate(dataset):
      table_idx = pc % table_size
      model = mdl_table[table_idx]
      optimizer = optim_table[table_idx]

      optimizer.zero_grad()
      output = model(data)
      loss = model.loss(output, label)
      loss.backward()
      optimizer.step()

      predicted_idx = torch.argmax(output)

      if label == predicted_idx: correct += 1
      total += 1

      if total % 100 == 0:
        mpkis += [1000.0* (total - correct) / (inst_count)]
        prev_correct = correct
        prec_inst_count = inst_count

      if (idx > num_samples):
        break
  finally:
    results[pid] = ((total - correct), inst_count) 
    print("{} ({}): miss = {} / {}; acc = {:0.2f}%; missPerKI = {:0.3f}".format(
      dataset, inst_count, (total - correct), total, (correct/total)*100.0, (1000.0 * (total - correct)) / inst_count))
    plt.plot(mpkis)
    plt.ylabel('Temporal missPerKI')
    plt.xlabel('Num predictions * 1000')
    print("bpred = ", mpkis)
    plt.show()
    plt.savefig(str(dataset) + '_' + str(num_samples) + '_' + str(table_size) + '_' + str(bhr_len) + '_' + str(lr) + 'bpred.png')

if __name__ == '__main__':
  NUM_SAMPLES = 10000
  TABLE_SIZE = 512
  BHR_LEN = 16
  LR = 0.15

  print("\nSamples={}; TABLE_SIZE={}; BHR_LEN={}; LR={}\n".format(
    NUM_SAMPLES, TABLE_SIZE, BHR_LEN, LR))

  jobs = []
  results = mp.Manager().dict()

  for i in range(1, len(sys.argv)):
    p = mp.Process(target=train, args=(i, sys.argv[i], LR, BHR_LEN, TABLE_SIZE, NUM_SAMPLES, results))
    jobs.append(p)
    p.start()

  for p in jobs:
    p.join()

  miss = 0
  inst_count = 0
  for key in results.keys():
    miss += results[key][0]
    inst_count += results[key][1]

  print("Samples={}; TABLE_SIZE={}; BHR_LEN={}; LR={}; Total missPerKI = {:0.3f}".format(
    NUM_SAMPLES, TABLE_SIZE, BHR_LEN, LR, (1000.0 * miss) / inst_count))
