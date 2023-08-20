import os
import time
import numpy as np
import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.transforms import ToPILImage
from torchvision.utils import save_image, make_grid
# import torch.nn.init as init
from tqdm import tqdm

from models.modules.pseudo_gt import expand_area
from models.loss import AnnealingComposePGT

from training.utils import plot_curves

class Solver():
    def __init__(self, config, args, inference=False):
        # self.load_folder = args.load_folder
        # self.save_folder = args.save_folder
        # self.vis_folder = os.path.join(args.save_folder, 'visualization')
        # if not os.path.exists(self.vis_folder):
        #     os.makedirs(self.vis_folder)
        self.vis_dest = args.o
        self.vis_freq = config.LOG.VIS_FREQ
        self.save_freq = config.LOG.SAVE_FREQ

        # Data & PGT
        self.img_size = config.DATA.IMG_SIZE
        self.margins = {'eye':config.PGT.EYE_MARGIN,
                        'lip':config.PGT.LIP_MARGIN}
        self.pgt_annealing = config.PGT.ANNEALING
        if self.pgt_annealing:
            self.pgt_maker = AnnealingComposePGT(self.margins, 
                config.PGT.SKIN_ALPHA_MILESTONES, config.PGT.SKIN_ALPHA_VALUES,
                config.PGT.EYE_ALPHA_MILESTONES, config.PGT.EYE_ALPHA_VALUES,
                config.PGT.LIP_ALPHA_MILESTONES, config.PGT.LIP_ALPHA_VALUES
            )
        else:
            self.pgt_maker = ComposePGT(self.margins, 
                config.PGT.SKIN_ALPHA,
                config.PGT.EYE_ALPHA,
                config.PGT.LIP_ALPHA
            )
        self.pgt_maker.eval()

        # Hyper-param
        self.num_epochs = config.TRAINING.NUM_EPOCHS

        self.device = args.device
        # self.keepon = args.keepon
        # self.logger = logger
        # self.build_model()
        super(Solver, self).__init__()


    def gen_pgt(self,data_loader):
        self.len_dataset = len(data_loader)
        with tqdm(data_loader, desc="training") as pbar:
            for self.epoch in range(1, self.num_epochs + 1):
                for step, (source, reference) in enumerate(pbar):
                    image_s, image_r = source[0], reference[0] # (b, c, h, w)
                    mask_s_full, mask_r_full = source[1], reference[1] # (b, c', h, w) 
                    diff_s, diff_r = source[2], reference[2] # (b, 136, h, w)
                    lms_s, lms_r = source[3], reference[3] # (b, K, 2)
                    pgt_A = self.pgt_maker(image_s, image_r, mask_s_full, mask_r_full, lms_s, lms_r)
                    # pgt_B = self.pgt_maker(image_r, image_s, mask_r_full, mask_s_full, lms_r, lms_s)
                    self.vis_train([pgt_A.detach().cpu()])
        # if (self.epoch) % self.vis_freq == 0:
            # self.vis_train([pgt_A.detach().cpu()])

    def de_norm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)
    
    def vis_train(self, img_train_batch):
        # saving training results
        img_train_batch = torch.cat(img_train_batch, dim=3)
        # i = 0
        save_path=self.vis_dest
        vis_image = make_grid(self.de_norm(img_train_batch), 1)
        save_image(vis_image, save_path) #, normalize=True)
        # i+=1
        # for im in img_train_batch:
        #     # save_path = os.path.join(self.vis_folder, 'epoch_{:d}_result{}.png'.format(self.epoch,i))
        #     save_pa
        #     vis_image = make_grid(self.de_norm(im), 1)
        #     save_image(vis_image, save_path) #, normalize=True)
        #     i+=1
