import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from training.config import get_config
from training.preprocess import PreProcess

class MakeupDataset(Dataset):
    def __init__(self, config=None, args=None):
        super(MakeupDataset, self).__init__()
        # if config is None:
        #     config = get_config()
        # self.root = config.DATA.PATH
        # with open(os.path.join(config.DATA.PATH, 'makeup.txt'), 'r') as f:
        #     self.makeup_names = [name.strip() for name in f.readlines()]
        # with open(os.path.join(config.DATA.PATH, 'non-makeup.txt'), 'r') as f:
        #     self.non_makeup_names = [name.strip() for name in f.readlines()]
        self.source_filename = args.s
        self.reference_filename = args.r
        self.preprocessor = PreProcess(config, need_parser=False)
        self.img_size = config.DATA.IMG_SIZE

    def load_from_file(self, img_name):
        # image = Image.open(os.path.join(self.root, 'images', img_name)).convert('RGB')
        # mask = self.preprocessor.load_mask(os.path.join(self.root, 'segs', img_name))
        # base_name = os.path.splitext(img_name)[0]
        # lms = self.preprocessor.load_lms(os.path.join(self.root, 'lms', f'{base_name}.npy'))
        image = Image.open(img_name)
        base_filename = img_name.split(".")[0]
        lms = self.preprocessor.load_lms(base_filename+".npy")
        mask = self.preprocessor.load_mask(img_name.split(".")[0]+"_segs."+img_name.split(".")[1])
        return self.preprocessor.process(image, mask, lms)
    
    def __len__(self):
        return 1
    #     return max(len(self.makeup_names), len(self.non_makeup_names))

    def __getitem__(self, index):
        # idx_s = torch.randint(0, len(self.non_makeup_names), (1, )).item()
        # idx_r = torch.randint(0, len(self.makeup_names), (1, )).item()
        # name_s = self.non_makeup_names[idx_s]
        # name_r = self.makeup_names[idx_r]
        source = self.load_from_file(self.source_filename)
        reference = self.load_from_file(self.reference_filename)
        return source, reference

def get_loader(config):
    dataset = MakeupDataset(config)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=config.DATA.BATCH_SIZE,
                            num_workers=config.DATA.NUM_WORKERS)
    return dataloader


if __name__ == "__main__":
    dataset = MakeupDataset()
    dataloader = DataLoader(dataset, batch_size=1, num_workers=16)
    for e in range(10):
        for i, (point_s, point_r) in enumerate(dataloader):
            pass