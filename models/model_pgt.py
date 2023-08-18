from .modules.histogram_matching import histogram_matching
from .modules.pseudo_gt import fine_align, expand_area, mask_blur


class ComposePGT():
    def __init__(self,skin_alpha_milestones, skin_alpha_values,eye_alpha_milestones, eye_alpha_values,lip_alpha_milestones, lip_alpha_values):
        super(ComposePGT, self).__init__()
        self.margins = margins
        self.skin_alpha_fn = LinearAnnealingFn(skin_alpha_milestones, skin_alpha_values)
        self.eye_alpha_fn = LinearAnnealingFn(eye_alpha_milestones, eye_alpha_values)
        self.lip_alpha_fn = LinearAnnealingFn(lip_alpha_milestones, lip_alpha_values)

        self.t = 0
        self.blend_alphas = {}
        self.step()

    def step(self):
        self.t += 1
        self.blend_alphas['skin'] = self.skin_alpha_fn(self.t)
        self.blend_alphas['eye'] = self.eye_alpha_fn(self.t)
        self.blend_alphas['lip'] = self.lip_alpha_fn(self.t)
    
    def forward(self, sources, targets, mask_srcs, mask_tars, lms_srcs,lms_tars):
        pgts = []
        for source, target, mask_src, mask_tar, lms_src, lms_tar in zip(sources, targets, mask_srcs, mask_tars, lms_srcs, lms_tars):
            pgt = generate_pgt(source, target, mask_src, mask_tar, lms_src, lms_tar,self.margins, self.blend_alphas)
            pgts.append(pgt)
        return pgts 

def norm(x: torch.Tensor):
    return x * 2 - 1

def de_norm(x: torch.Tensor):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def masked_his_match(image_s, image_r, mask_s, mask_r):
    '''
    image: (3, h, w)
    mask: (1, h, w)
    '''
    index_tmp = torch.nonzero(mask_s)
    x_A_index = index_tmp[:, 1]
    y_A_index = index_tmp[:, 2]
    index_tmp = torch.nonzero(mask_r)
    x_B_index = index_tmp[:, 1]
    y_B_index = index_tmp[:, 2]

    image_s = (de_norm(image_s) * 255) #[-1, 1] -> [0, 255]
    image_r = (de_norm(image_r) * 255)
    
    source_masked = image_s * mask_s
    target_masked = image_r * mask_r
    
    source_match = histogram_matching(
                source_masked, target_masked,
                [x_A_index, y_A_index, x_B_index, y_B_index])
    source_match = source_match.to(image_s.device)
    
    return norm(source_match / 255) #[0, 255] -> [-1, 1]


def generate_pgt(image_s, image_r, mask_s, mask_r, lms_s, lms_r, margins, blend_alphas, img_size=None):
        """
        input_data: (3, h, w)
        mask: (c, h, w), lip, skin, left eye, right eye
        """
        if img_size is None:
            img_size = image_s.shape[1]
        pgt = image_s.detach().clone()

        # skin match
        skin_match = masked_his_match(image_s, image_r, mask_s[1:2], mask_r[1:2])
        pgt = (1 - mask_s[1:2]) * pgt + mask_s[1:2] * skin_match

        # lip match
        lip_match = masked_his_match(image_s, image_r, mask_s[0:1], mask_r[0:1])
        pgt = (1 - mask_s[0:1]) * pgt + mask_s[0:1] * lip_match

        # eye match
        mask_s_eye = expand_area(mask_s[2:4].sum(dim=0, keepdim=True), margins['eye']) * mask_s[1:2]
        mask_r_eye = expand_area(mask_r[2:4].sum(dim=0, keepdim=True), margins['eye']) * mask_r[1:2]
        eye_match = masked_his_match(image_s, image_r, mask_s_eye, mask_r_eye)
        mask_s_eye_blur = mask_blur(mask_s_eye, blur_size=5, mode='valid')
        pgt = (1 - mask_s_eye_blur) * pgt + mask_s_eye_blur * eye_match

        # tps align
        pgt = fine_align(img_size, lms_r, lms_s, image_r, pgt, mask_r, mask_s, margins, blend_alphas)
        return pgt
