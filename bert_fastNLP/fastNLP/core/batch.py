import numpy as np
import torch


class Batch(object):
    """Batch is an iterable object which iterates over mini-batches.

        Example::

            for batch_x, batch_y in Batch(data_set, batch_size=16, sampler=SequentialSampler()):
                # ...

    :param DataSet dataset: a DataSet object
    :param int batch_size: the size of the batch
    :param Sampler sampler: a Sampler object
    :param bool as_numpy: If True, return Numpy array. Otherwise, return torch tensors.

    """

    def __init__(self, dataset, batch_size, sampler, as_numpy=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampler = sampler
        self.as_numpy = as_numpy
        self.idx_list = None
        self.curidx = 0
        self.num_batches = len(dataset)//batch_size + int(len(dataset)%batch_size!=0)

    def __iter__(self):
        self.idx_list = self.sampler(self.dataset)
        self.curidx = 0
        self.lengths = self.dataset.get_length()
        return self

    def __next__(self):
        if self.curidx >= len(self.idx_list):
            raise StopIteration
        else:
            endidx = min(self.curidx + self.batch_size, len(self.idx_list))
            batch_x, batch_y = {}, {}

            indices = self.idx_list[self.curidx:endidx]

            for field_name, field in self.dataset.get_all_fields().items():
                if field.is_target or field.is_input:
                    batch = field.get(indices)
                    if not self.as_numpy:
                        batch = to_tensor(batch, field.dtype)
                    if field.is_target:
                        batch_y[field_name] = batch
                    if field.is_input:
                        batch_x[field_name] = batch

            self.curidx = endidx

            return batch_x, batch_y

    def __len__(self):
        return self.num_batches


def to_tensor(batch, dtype):
    if dtype in (int, np.int8, np.int16, np.int32, np.int64):
        batch = torch.LongTensor(batch)
    if dtype in (float, np.float32, np.float64):
        batch = torch.FloatTensor(batch)
    return batch
