import numpy as np
from torch.utils.data.sampler import BatchSampler

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size
    
# first sample class, then item with replacement
# num_classes = fg_num_classes + 1
class BalancedStatisticSampler(BatchSampler):
    def __init__(self, labels, num_classes, batch_size):
        self.labels = np.array(labels)
        self.n_classes = num_classes
        self.batch_size = batch_size
        
        self.n_dataset = len(self.labels)
        self.labels_set = list(set(self.labels))
        self.indices_table = {
            label: np.where(self.labels == label)[0] for label in self.labels_set
        }
        
    def __iter__(self):
        self.count = 0
        
        while self.count + self.batch_size < self.n_dataset:
            indices = []
            
            for i in range(self.batch_size):
                curr_cls = np.random.randint(0, self.n_classes)
                index = np.random.randint(0, len(self.indices_table[curr_cls]))
                indices.append(self.indices_table[curr_cls][index])
            
            yield indices
            
            self.count += self.batch_size
        
    def __len__(self):
        return self.n_dataset // self.batch_size
    