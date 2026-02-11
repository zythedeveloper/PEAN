import torch
import os
import argparse
import yaml
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from interfaces import base
from dataset.dataset import LRLPRDataset, alignCollate_real_sequence
from utils.util import str_filt
from utils.metrics import get_str_list
from easydict import EasyDict
import warnings
warnings.filterwarnings("ignore")

class LRLPRTester(base.TextBase):
    def __init__(self, config, args):
        super(LRLPRTester, self).__init__(config, args)
        self.config = config
        self.align_collate = alignCollate_real_sequence

    def test_lrlpr(self):
        # 1. Initialize Models
        model_dict = self.generator_init()
        model = model_dict['model'].eval()
        
        # Prior generator for PEAN attention
        recognizer_prior = self.PARSeq_init().eval()
        
        # Performance recognizer (Aster)
        aster, aster_info = self.Aster_init()
        aster.eval()

        # 2. Setup Dataset (is_training=True loads GT HR frames)
        dataset = LRLPRDataset(root_dir=self.args.test_data_dir, is_training=True)
        test_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=self.config.TRAIN.workers,
            collate_fn=self.align_collate(
                imgH=self.config.TRAIN.height, 
                imgW=self.config.TRAIN.width, 
                down_sample_scale=self.config.TRAIN.down_sample_scale,
                mask=self.mask
            )
        )

        metrics = {'psnr': [], 'ssim': [], 'n_correct': 0, 'total': 0}
        
        print(f"Validating PEAN on LRLPR Train Split: {len(dataset)} tracks")

        with torch.no_grad():
            for i, (hr_batch, lr_batch, label_gt, metadata) in enumerate(tqdm(test_loader)):
                sr_frames = []
                lr_seq = lr_batch.squeeze(0) 
                hr_seq = hr_batch.squeeze(0)
                
                # Now num_frames will correctly be 5
                num_frames = lr_seq.size(0)
                
                for f_idx in range(num_frames):
                    lr_img = lr_seq[f_idx]
                    hr_img = hr_seq[f_idx]

                    lr_tensor = lr_img.unsqueeze(0).to(self.device) 
                    hr_tensor = hr_img.unsqueeze(0).to(self.device)
                    
                    # A. Generate Prior (Pseudo-labels)
                    parseq_input = self.parse_parseq_data(lr_img)
                    pseudo_output = recognizer_prior(parseq_input)
                    label_vecs = pseudo_output.softmax(-1)

                    # B. Inference
                    sr_tensor, _ = model(lr_tensor, label_vecs)
                    sr_frames.append(sr_tensor)

                    # C. Calculate Image Metrics (Per-frame)
                    # Note: tensors are typically in [-1, 1] range after transform
                    psnr = self.cal_psnr(sr_tensor, hr_tensor)
                    ssim = self.cal_ssim(sr_tensor, hr_tensor)
                    metrics['psnr'].append(psnr.item())
                    metrics['ssim'].append(ssim.item())

                # 3. Multi-Frame Fusion for Recognition
                fused_sr = torch.stack(sr_frames).mean(dim=0)

                # 4. Accuracy Assessment
                aster_dict = self.parse_aster_data(fused_sr[:, :3, :, :])
                aster_output = aster(aster_dict)
                pred_rec = aster_output['output']['pred_rec']
                
                # Decode and Filter
                pred_str, _ = get_str_list(pred_rec, aster_dict['rec_targets'], dataset=aster_info)
                
                if str_filt(pred_str[0], 'lower') == str_filt(label_gt[0], 'lower'):
                    metrics['n_correct'] += 1
                
                metrics['total'] += 1

        # 5. Final Report
        avg_psnr = np.mean(metrics['psnr'])
        avg_ssim = np.mean(metrics['ssim'])
        accuracy = metrics['n_correct'] / metrics['total']

        print("\n" + "="*30)
        print(f"LRLPR EVALUATION RESULTS")
        print("-" * 30)
        print(f"PSNR:     {avg_psnr:.2f}")
        print(f"SSIM:     {avg_ssim:.4f}")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print("="*30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--pre_training', action='store_true', default=False)
    parser.add_argument('--test_data_dir', type=str, default='/root/dataset/TextZoom/test/medium', help='')
    parser.add_argument('--batch_size', type=int, default=None, help='')
    parser.add_argument('--resume', type=str, default=None, help='')
    parser.add_argument('--vis_dir', type=str, default=None, help='')
    parser.add_argument('--rec', default='aster', choices=['aster', 'moran', 'crnn'])
    parser.add_argument('--STN', action='store_true', default=False, help='')
    parser.add_argument('--syn', action='store_true', default=False, help='use synthetic LR')
    parser.add_argument('--mixed', action='store_true', default=False, help='mix synthetic with real LR')
    parser.add_argument('--mask', action='store_true', default=False, help='')
    parser.add_argument('--gradient', action='store_true', default=False, help='')
    parser.add_argument('--hd_u', type=int, default=32, help='')
    parser.add_argument('--srb', type=int, default=5, help='')
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--demo_dir', type=str, default='')
    parser.add_argument('--prior_dim', type=int, default=1024, help='')
    parser.add_argument('--dec_num_heads', type=int, default=16, help='')
    parser.add_argument('--dec_mlp_ratio', type=int, default=4, help='')
    parser.add_argument('--dec_depth', type=int, default=1, help='')
    parser.add_argument('--max_gen_perms', type=int, default=1, help='')
    parser.add_argument('--rotate_train', type=float, default=0., help='')
    parser.add_argument('--perm_forward', action='store_true', default=False, help='')
    parser.add_argument('--perm_mirrored', action='store_true', default=False, help='')
    parser.add_argument('--dropout', type=float, default=0.1, help='')

    args = parser.parse_args()
    config_path = os.path.join('config', 'super_resolution.yaml')
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    config = EasyDict(config)

    tester = LRLPRTester(config, args)
    tester.test_lrlpr()