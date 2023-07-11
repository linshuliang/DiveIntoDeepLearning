import torch
import torch.nn as nn

input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
print(input.size()) # [2, 4]

embed = nn.Embedding(10, 3)
print(embed.weight.size()) # [10, 3]

a = embed(input)
print(a.size()) # [2, 4, 3]