"""
Change built in dgl graph dataset into the format of YiTu_GNN.NDP
"""

import os
import argparse
import numpy as np
import scipy.sparse
from dgl.data.utils import get_download_dir

def convert_reddit_data(dataset, dir, out_folder, self_loop=False):
  """
  Load DGL graph dataset
  """
  self_loop_str = ""
  if self_loop:
    self_loop_str = "_self_loop"
  if dir==None:
    dir = ""
  extract_dir = os.path.join(dir, "{}{}".format(dataset, self_loop_str))

  coo_adj = scipy.sparse.load_npz(os.path.join(extract_dir, "{}{}_graph.npz"
                                     .format(dataset, self_loop_str)))

  reddit_data = np.load(os.path.join(extract_dir, "{}_data.npz".format(dataset)))
  features = reddit_data["feature"]
  labels = reddit_data["label"]
  node_types = reddit_data["node_types"]
  train_mask = (node_types == 1)
  val_mask = (node_types == 2)
  test_mask = (node_types == 3)

  scipy.sparse.save_npz(os.path.join(out_folder, 'adj.npz'), coo_adj)
  np.save(os.path.join(out_folder, 'feat.npy'), features)
  np.save(os.path.join(out_folder, 'labels.npy'), labels)
  np.save(os.path.join(out_folder, 'train.npy'), train_mask)
  np.save(os.path.join(out_folder, 'val.npy'), val_mask)
  np.save(os.path.join(out_folder, 'test.npy'), test_mask)

  print('Convert Finishes')


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Convert')

  parser.add_argument("--dataset", type=str, default=None,
                      help="DGL dataset name")
  parser.add_argument("--self-loop", dest='self_loop', action='store_true')
  parser.set_defaults(self_loop=False)
  parser.add_argument("--dir", type=str, default=None,
                      help="dataset dir")
  parser.add_argument("--out-dir", type=str, default=None,
                      help="saved dataset folder")
  args = parser.parse_args()

  if 'reddit' in args.dataset:
    print('Converting {} dataset...'.format(args.dataset))
    convert_reddit_data(args.dataset, args.dir, args.out_dir, args.self_loop)
  else:
    print('Unsupported Dataset.')
  
