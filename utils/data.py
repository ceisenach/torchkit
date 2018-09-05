import numpy as np
import torch
import copy

__all__ = ['MultiRingBuffer']

class MultiRingBuffer(object):
    """
    Ring buffer that supports multiple data of different widths
    """
    def __init__(self, experience_shapes, max_len,tensor_type=torch.FloatTensor):
        assert isinstance(experience_shapes, list)
        assert len(experience_shapes) > 0
        assert isinstance(experience_shapes[0], tuple)
        self.maxlen = max_len
        self.start = 0
        self.length = 0
        self.dataList = [tensor_type(max_len, *shape).zero_() for shape in experience_shapes]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return [data[(self.start + idx) % self.maxlen] for data in self.dataList]

    def append(self, *vs):
        """
        Append to buffer
        """
        vs = list(vs)
        for i, v in enumerate(vs):
            if isinstance(v, np.ndarray):
                vs[i] = torch.from_numpy(v)
        if self.length < self.maxlen:
            # Space Available
            self.length += 1
        elif self.length == self.maxlen:
            # No Space, remove the first item
            self.start = (self.start + 1) % self.maxlen
        else:
            # Should not happen
            raise RuntimeError()
        for data, v in zip(self.dataList, vs):
            data[(self.start + self.length - 1) % self.maxlen] = v.squeeze()

    def reset(self):
        """
        Clear replay buffer
        """
        self.start = 0
        self.length = 0

    def get_data(self):
        """
        Get all data in the buffer
        """
        if self.length < self.maxlen:
            return [data[0:self.length] for data in self.dataList]
        return self.dataList
