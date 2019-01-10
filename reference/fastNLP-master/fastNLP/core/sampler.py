from itertools import chain

import numpy as np
import torch


def convert_to_torch_tensor(data_list, use_cuda):
    """Convert lists into (cuda) Tensors.

    :param data_list: 2-level lists
    :param use_cuda: bool, whether to use GPU or not
    :return data_list: PyTorch Tensor of shape [batch_size, max_seq_len]
    """
    data_list = torch.Tensor(data_list).long()
    if torch.cuda.is_available() and use_cuda:
        data_list = data_list.cuda()
    return data_list


class BaseSampler(object):
    """The base class of all samplers.

        Sub-classes must implement the ``__call__`` method.
        ``__call__`` takes a DataSet object and returns a list of int - the sampling indices.
    """

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class SequentialSampler(BaseSampler):
    """Sample data in the original order.

    """
    def __call__(self, data_set):
        """

        :param DataSet data_set:
        :return result: a list of integers.
        """
        return list(range(len(data_set)))


class RandomSampler(BaseSampler):
    """Sample data in random permutation order.

    """
    def __call__(self, data_set):
        """

            :param DataSet data_set:
            :return result: a list of integers.
        """
        return list(np.random.permutation(len(data_set)))


class BucketSampler(BaseSampler):
    """

        :param int num_buckets: the number of buckets to use.
        :param int batch_size: batch size per epoch.
        :param str seq_lens_field_name: the field name indicating the field about sequence length.

    """
    def __init__(self, num_buckets=10, batch_size=32, seq_lens_field_name='seq_lens'):
        self.num_buckets = num_buckets
        self.batch_size = batch_size
        self.seq_lens_field_name = seq_lens_field_name

    def __call__(self, data_set):

        seq_lens = data_set.get_all_fields()[self.seq_lens_field_name].content
        total_sample_num = len(seq_lens)

        bucket_indexes = []
        num_sample_per_bucket = total_sample_num // self.num_buckets
        for i in range(self.num_buckets):
            bucket_indexes.append([num_sample_per_bucket * i, num_sample_per_bucket * (i + 1)])
        bucket_indexes[-1][1] = total_sample_num

        sorted_seq_lens = list(sorted([(idx, seq_len) for
                                       idx, seq_len in zip(range(total_sample_num), seq_lens)],
                                      key=lambda x: x[1]))

        batchs = []

        left_init_indexes = []
        for b_idx in range(self.num_buckets):
            start_idx = bucket_indexes[b_idx][0]
            end_idx = bucket_indexes[b_idx][1]
            sorted_bucket_seq_lens = sorted_seq_lens[start_idx:end_idx]
            left_init_indexes.extend([tup[0] for tup in sorted_bucket_seq_lens])
            num_batch_per_bucket = len(left_init_indexes) // self.batch_size
            np.random.shuffle(left_init_indexes)
            for i in range(num_batch_per_bucket):
                batchs.append(left_init_indexes[i * self.batch_size:(i + 1) * self.batch_size])
            left_init_indexes = left_init_indexes[num_batch_per_bucket * self.batch_size:]
        if (left_init_indexes) != 0:
            batchs.append(left_init_indexes)
        np.random.shuffle(batchs)

        return list(chain(*batchs))


def simple_sort_bucketing(lengths):
    """

    :param lengths: list of int, the lengths of all examples.
    :return data: 2-level list
            ::

                [
                    [index_11, index_12, ...],  # bucket 1
                    [index_21, index_22, ...],  # bucket 2
                    ...
                ]

    """
    lengths_mapping = [(idx, length) for idx, length in enumerate(lengths)]
    sorted_lengths = sorted(lengths_mapping, key=lambda x: x[1])
    # TODO: need to return buckets
    return [idx for idx, _ in sorted_lengths]


def k_means_1d(x, k, max_iter=100):
    """Perform k-means on 1-D data.

    :param x: list of int, representing points in 1-D.
    :param k: the number of clusters required.
    :param max_iter: maximum iteration
    :return centroids: numpy array, centroids of the k clusters
            assignment: numpy array, 1-D, the bucket id assigned to each example.
    """
    sorted_x = sorted(list(set(x)))
    x = np.array(x)
    if len(sorted_x) < k:
        raise ValueError("too few buckets")
    gap = len(sorted_x) / k

    centroids = np.array([sorted_x[int(x * gap)] for x in range(k)])
    assign = None

    for i in range(max_iter):
        # Cluster Assignment step
        assign = np.array([np.argmin([np.absolute(x_i - x) for x in centroids]) for x_i in x])
        # Move centroids step
        new_centroids = np.array([x[assign == k].mean() for k in range(k)])
        if (new_centroids == centroids).all():
            centroids = new_centroids
            break
        centroids = new_centroids
    return np.array(centroids), assign


def k_means_bucketing(lengths, buckets):
    """Assign all instances into possible buckets using k-means, such that instances in the same bucket have similar lengths.

    :param lengths: list of int, the length of all samples.
    :param buckets: list of int. The length of the list is the number of buckets. Each integer is the maximum length
        threshold for each bucket (This is usually None.).
    :return data: 2-level list
            ::

                [
                    [index_11, index_12, ...],  # bucket 1
                    [index_21, index_22, ...],  # bucket 2
                    ...
                ]

    """
    bucket_data = [[] for _ in buckets]
    num_buckets = len(buckets)
    _, assignments = k_means_1d(lengths, num_buckets)

    for idx, bucket_id in enumerate(assignments):
        if buckets[bucket_id] is None or lengths[idx] <= buckets[bucket_id]:
            bucket_data[bucket_id].append(idx)
    return bucket_data
