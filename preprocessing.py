import os.path as osp
import os
import numpy as np
import shutil
import argparse



def mkdirs(d, replace = True):
    if not osp.exists(d):
        os.makedirs(d)
    elif replace:
        shutil.rmtree(d)
        os.makedirs(d)


def preprocessing_gt(seq_root, trainer = ['train']):

    _label_root = seq_root + '/trackers_gt_t'
    mkdirs(_label_root)

    for tr in trainer:
        print(tr)
        label_root = _label_root + '/' + tr
        mkdirs(label_root)
        seq_root_tr = (osp.join(seq_root, tr))
        seqs =[s for s in os.listdir(seq_root_tr)  if not s.startswith('.') and "gt_t" not in s]


        for seq in seqs:
            print(seq)
            seq_info = open(osp.join(seq_root_tr, seq, 'seqinfo.ini')).read()
            seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
            seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

            gt_txt = osp.join(seq_root_tr, seq, 'gt', 'gt.txt')
            gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
            idx = np.lexsort(gt.T[:2, :])
            gt = gt[idx, :]

            seq_label_root = osp.join(label_root, seq, 'img1')
            mkdirs(seq_label_root, replace = False)

            for fid, tid, x, y, w, h, mark, cls, vis in gt:
                if mark == 0 or not cls == 1:
                    continue
                fid = int(fid)
                tid = int(tid)
                x += w / 2
                y += h / 2
                label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(tid))
                label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:6f} {:6f}\n'.format(
                    fid, x / seq_width, y / seq_height, w / seq_width, h / seq_height, vis, seq_width, seq_height)
                with open(label_fpath, 'a') as f:
                    f.write(label_str)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess ground truth tracking data.")
    parser.add_argument("--seq_root", type=str, required=True, help="Root directory containing sequence data.")
    parser.add_argument("--train", action='store_true', help="Include train set in preprocessing.")
    parser.add_argument("--val", action='store_true', help="Include validation set in preprocessing.")
    # parser.add_argument("--test", action='store_true', help="Include test set in preprocessing.")

    args = parser.parse_args()

    # Create trainer list based on specified arguments
    trainer = []
    if args.train:
        trainer.append('train')
    if args.val:
        trainer.append('val')
    # if args.test:
    #     trainer.append('test')

    # Run preprocessing
    preprocessing_gt(args.seq_root, trainer=trainer)