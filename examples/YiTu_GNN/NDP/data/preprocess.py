"""
Preprocess dataset to fit the input
"""

import numpy as np
import scipy.sparse
import os
import sys
import argparse

def pp2adj(filepath, is_direct=True, delimiter='\t',
           outfile=None):
  """
  Convert (vertex vertex) tuple into numpy adj matrix
  adj matrix will be returned.
  If outfile is provided, also save it.
  """
  pp = np.loadtxt(filepath, delimiter=delimiter)
  src_node = pp[:,0].astype(np.int)
  dst_node = pp[:,1].astype(np.int)
  max_nid = max(np.max(src_node), np.max(dst_node))
  print('max_nid:{}'.format(max_nid))
  min_nid = min(np.min(src_node), np.min(dst_node))
  print('min_nid:{}'.format(min_nid))

  # get vertex and ege num info
  vnum = max_nid - min_nid + 1
  enum = len(src_node) if is_direct else len(src_node) * 2
  print('vertex#: {} edge#: {}'.format(vnum, enum))

  # scale node id from 0
  src_node -= min_nid
  dst_node -= min_nid

  # make coo sparse adj matrix
  if not is_direct:
    src_node, dst_node = np.concatenate((src_node, dst_node)), \
                         np.concatenate((dst_node, src_node))
  edge_weight = np.ones(enum, dtype=np.int)
  coo_adj = scipy.sparse.coo_matrix(
    (edge_weight, (src_node, dst_node)),
    shape=(vnum, vnum)
  )
  # output to file
  if outfile is not None:
    scipy.sparse.save_npz(outfile, coo_adj)
  return coo_adj


def random_feature(vnum, feat_size, outfile=None):
  """
  Generate random features using numpy
  Params:
    vnum:       feature num (aka. vertex num)
    feat_size:  feature dimension 
    outfile:    save to the file if provided
  Returns:
    numpy array obj with shape of [vnum, feat_size]
  """
  feat_mat = np.random.random((vnum, feat_size)).astype(np.float32)
  if outfile:
    np.save(outfile, feat_mat)
  return feat_mat


def random_label(vnum, class_num, outfile=None):
  """
  Generate random labels from 0 - class_num for each node
  Params:
    vnum:       total node num
    class_num:  number of classes, start from 0
    outfile:    save to the file if provided
  Returns:
    numpy array obj with shape of (vnum,).
    Each element denotes corresponding nodes labels
  """
  labels = np.random.randint(class_num, size=vnum)
  if outfile:
    np.save(outfile, labels)
  return labels


def split_dataset(vnum, outdir=None):
  """
  Split dataset to train/val/test.
  train:val:test = 6.5:1:1.5 - similar to reddit
  if outdir is provided:
    save as outdir/train.npy,
            outdir/val.npy,
            outdir/test.npy
  Return:
    3 ndarrays with train, val, test mask.
    All of them is of (vnum,) size with 0 or 1 indicator. 
  """
  nids = np.arange(vnum)
  np.random.shuffle(nids)
  train_len = int(vnum * 0.65)
  val_len = int(vnum * 0.1)
  test_len = vnum - train_len - val_len
  # train mask
  train_mask = np.zeros(vnum, dtype=np.int)
  train_mask[nids[0:train_len]] = 1
  # val mask
  val_mask = np.zeros(vnum, dtype=np.int)
  val_mask[nids[train_len:train_len + val_len]] = 1
  # test mask
  test_mask = np.zeros(vnum, dtype=np.int)
  test_mask[nids[-test_len:]] = 1
  # save
  if outdir is not None:
    np.save(os.path.join(outdir, 'train.npy'), train_mask)
    np.save(os.path.join(outdir, 'val.npy'), val_mask)
    np.save(os.path.join(outdir, 'test.npy'), test_mask)
  return train_mask, val_mask, test_mask


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Preprocess')

  parser.add_argument("--dataset", type=str, default=None,
                      help="dataset dir")

  parser.add_argument("--ppfile", type=str, default=None,
                      help='point-to-point graph filename')
  parser.add_argument("--directed", dest="directed", action='store_true')
  parser.set_defaults(directed=False)

  parser.add_argument("--gen-feature", dest='gen_feature', action='store_true')
  parser.set_defaults(gen_feature=False)
  parser.add_argument("--feat-size", type=int, default=600,
                      help='generated feature size if --gen-feature is specified')
  
  parser.add_argument("--gen-label", dest='gen_label', action='store_true')
  parser.set_defaults(gen_label=False)
  parser.add_argument("--class-num", type=int, default=60,
                      help='generated class number if --gen-label is specified')
  
  parser.add_argument("--gen-set", dest='gen_set', action='store_true')

  parser.set_defaults(gen_set=False)
  args = parser.parse_args()

  if not os.path.exists(args.dataset):
    print('{}: No such a dataset folder'.format(args.dataset))
    sys.exit(-1)
  
  # generate adj
  adj_file = os.path.join(args.dataset, 'adj.npz')
  if args.ppfile is not None:
    print('Generating adj matrix in: {}...'.format(adj_file))
    adj = pp2adj(
      os.path.join(args.dataset, args.ppfile),
      is_direct=args.directed,
      outfile=adj_file
    )
  else:
    adj = scipy.sparse.load_npz(adj_file)
  vnum = adj.shape[0]
  del adj

  # generate features
  feat_file = os.path.join(args.dataset, 'feat.npy')
  if args.gen_feature:
    print('Generating random features (size: {}) in: {}...'
          .format(args.feat_size, feat_file))
    feat = random_feature(vnum, args.feat_size, 
                          outfile=feat_file)
  
  # generate labels
  label_file = os.path.join(args.dataset, 'labels.npy')
  if args.gen_label:
    print('Generating labels (class num: {}) in: {}...'
          .format(args.class_num, feat_file))
    labels = random_label(vnum, args.class_num,
                          outfile=label_file)
  
  # generate train/val/test set
  if args.gen_set:
    print('Generating train/val/test masks in: {}...'
          .format(args.dataset))
    split_dataset(vnum, outdir=args.dataset)
  
  print('Done.')