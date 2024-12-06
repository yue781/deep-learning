# same size with input matrix (for adding with input matrix)
self.encoding = torch.zeros(max_len, d_model)
self.encoding.requires_grad = False  # we don't need to compute gradient

pos = torch.arange(0, max_len)
pos = pos.float().unsqueeze(dim=1)
# 1D => 2D unsqueeze to represent word's position

self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

# 这两行代码的作用是通过正弦和余弦函数计算每个位置的编码，并将它们分别赋值给位置编码张量 self.encoding 的偶数和奇数维度。这样做的好处是：
# 位置编码的每个维度具有不同的频率：正弦和余弦函数的不同周期使得位置编码能够涵盖多个尺度的位置信息。
# 正弦和余弦函数互补：正弦和余弦的组合保证了位置编码能够捕捉到序列中的相对位置信息，并且它们的周期性特征使得模型能够有效地识别不同位置之间的关系。
# 避免在相邻位置编码之间产生重复：通过为每个维度使用不同的频率，确保每个位置编码在高维空间中都有唯一性

# self.encoding 是一个二维张量，形状为 (max_len, d_model)，用来存储位置编码。
# [:, 0::2] 是对 self.encoding 张量的切片操作：
# : 表示选择所有的行（即每个位置的编码）。
# 0::2 表示选择从第 0 列开始，每隔 2 列取一次，即选择所有偶数索引的列。也就是 self.encoding 中的第 0, 2, 4, 6, ..., d_model-2 维度的列。
# 这些列对应的是位置编码的奇数维度，将使用正弦函数进行计算。

# pos 是一个张量，表示每个位置的索引，形状为 (max_len, 1)，每一行的元素表示当前位置的索引（从 0 到 max_len-1）。
# 例如，如果 max_len = 512，那么 pos 将是 [0, 1, 2, ..., 511] 的一列张量。

# _2i 是一个张量，表示位置编码的每个维度的索引。它是一个从 0 开始的数字列表，每隔 2 个元素取一次（即 0, 2, 4, ..., d_model-2）。
# _2i = torch.arange(0, d_model, step=2).float()，表示从 0 开始，以 2 为步长，生成一个范围为 [0, 2, 4, ..., d_model-2] 的张量。
# 10000 ** (_2i / d_model)

# 这一部分计算了一个缩放因子，它控制不同维度的频率。
# _2i / d_model：对 _2i 张量的每个元素除以 d_model，得到一个比例因子。假设 d_model=512，_2i 会是 [0, 2, 4, ..., 510]，那么 _2i / d_model 会是 [0, 0.0039, 0.0078, ..., 0.9961]。
# 10000 ** (_2i / d_model)：计算 10000 的 (_2i / d_model) 次幂，从而得到不同频率的缩放因子。这样做的目的是让每个维度的编码具有不同的周期。对于小的维度索引（较小的 _2i），其周期会较长，而对于较大的维度索引，周期会较短。这有助于让模型能够学习到位置编码中不同频率的信息。
# torch.sin(pos / (10000 ** (_2i / d_model)))

# 最后，使用正弦函数对每个位置进行编码。
# 这部分的核心思想是：根据位置索引（pos）和缩放因子（10000 ** (_2i / d_model)），计算每个位置的编码。
# 对每个位置 pos 除以缩放因子 10000 ** (_2i / d_model)，然后使用正弦函数对结果进行计算。
# torch.sin(...) 会为每个位置 pos 和维度 _2i 计算正弦值，这样每个位置的编码就会根据维度不同而呈现不同的周期性变化。
