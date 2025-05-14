from unicore.data import NestedDictionaryDataset

class CustomNestedDictionaryDataset(NestedDictionaryDataset):
    def __init__(self, defn, valid_indices=None):
        super().__init__(defn)
        self.valid_indices = valid_indices
        
    def __getitem__(self, idx):
        if self.valid_indices is None:
            return super().__getitem__(idx)
        return super().__getitem__(self.valid_indices[idx])
        
    def __len__(self):
        # 计算有效数据的数量
        if self.valid_indices is None:
            return super().__len__()
        return len(self.valid_indices)