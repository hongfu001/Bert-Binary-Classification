import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class MyDataset(Dataset):
    def __init__(self, mode, subject, validation_ratio):
        super(MyDataset,self).__init__()
        self.mode = mode
        self.subject = subject

        raw_train_data = pd.read_json('warmup_ice.json', lines=True)
        raw_test_data = pd.read_json('test_data.json', lines=True)

        if subject == 'total':
            train_data, val_data = train_test_split(raw_train_data, test_size=validation_ratio, shuffle=True)
            test_data = raw_test_data
        else:
            unsafe_train = raw_train_data.loc[raw_train_data['label_sub'] == self.subject]
            others, safe_train = train_test_split(raw_train_data.loc[raw_train_data['label_sub'] == 0], test_size=len(unsafe_train), shuffle=True)
            unsafe_test = raw_test_data.loc[raw_test_data['label_sub'] == self.subject]
            others, safe_test = train_test_split(raw_test_data.loc[raw_test_data['label_sub'] == 0], test_size=len(unsafe_test), shuffle=True)

            


            raw1_train_data = pd.concat([safe_train, unsafe_train])
            test_data = pd.concat([safe_test, unsafe_test])

            train_data, val_data = train_test_split(raw1_train_data, test_size=validation_ratio, shuffle=True)


        if mode == 'train':
            self.dataset = train_data
        elif mode == 'validation':
            self.dataset = val_data
        elif mode == 'test':
            self.dataset = test_data
        else:
            print('error')
        
    
    def __getitem__(self, index):
        data = self.dataset.iloc[index]
#         # 取其推文，做个简单的数据清理
        source = data['text']
        label = data['label']
#         # 返回推文和label
        return source, label

    def __len__(self):
        return len(self.dataset)