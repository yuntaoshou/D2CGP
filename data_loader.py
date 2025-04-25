import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        DATASET_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
        }
        DATASET_MAP[args['dataset_name']]()

    def __init_mosi(self):
        with open(self.args['featurePath'], 'rb') as f:
            data = pickle.load(f)
        if 'use_bert' in self.args and self.args['use_bert']:
            self.text = data[self.mode]['text_bert'].astype(np.float32)
        else:
            self.text = data[self.mode]['text'].astype(np.float32)
        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)
        self.raw_text = data[self.mode]['raw_text']
        self.ids = data[self.mode]['id']

        self.labels = {
            'M': np.array(data[self.mode]['regression_labels']).astype(np.float32)
        }

        self.audio[self.audio == -np.inf] = 0

        if 'need_normalized' in self.args and self.args['need_normalized']:
            self.__normalize()
    
    def __init_mosei(self):
        return self.__init_mosi()

    def __normalize(self):
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))
        self.vision = np.mean(self.vision, axis=0, keepdims=True)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)

        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0

        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))

    def __len__(self):
        return len(self.labels['M'])

    def get_seq_len(self):
        if 'use_bert' in self.args and self.args['use_bert']:
            return (self.text.shape[2], self.audio.shape[1], self.vision.shape[1])
        else:
            return (self.text.shape[1], self.audio.shape[1], self.vision.shape[1])

    def get_feature_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def __getitem__(self, index):
        sample = {
            'raw_text': self.raw_text[index],
            'text': torch.Tensor(self.text[index]), 
            'audio': torch.Tensor(self.audio[index]),
            'vision': torch.Tensor(self.vision[index]),
            'index': index,
            'id': self.ids[index],
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
        } 
        return sample

def MMDataLoader(args, num_workers):

    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    if 'seq_lens' in args:
        args['seq_lens'] = datasets['train'].get_seq_len() 

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args['batch_size'],
                       num_workers=num_workers,
                       shuffle=True)
        for ds in datasets.keys()
    }
    
    return dataLoader
