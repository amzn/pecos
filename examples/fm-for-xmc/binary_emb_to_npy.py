import argparse
import os
import numpy as np
import struct

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("emb_dir", help="Directory of FM embeddings", type=str)
    return parser.parse_args()

def test_eof(f):
    remaining_bytes = 0
    while True:
        if f.read(1) != b'':
            remaining_bytes += 1
        else:
            break
    if remaining_bytes != 0:
        raise Exception(f"Expected to reach EOF but got {remaining_bytes} bytes left.")

def construct_emb_with_bias(embs_dir, save=True, verbose=False):
    with open(os.path.join(embs_dir, "X.emb"), 'rb') as f:
        r = struct.unpack('i', f.read(4))[0]
        c = struct.unpack('i', f.read(4))[0]
        X_embs = np.reshape(np.fromfile(f, dtype=np.float32), (r, c))
        
        test_eof(f)
    
    with open(os.path.join(embs_dir, "X.bias"), 'rb') as f:
        r = struct.unpack('i', f.read(4))[0]
        X_bias = np.reshape(np.fromfile(f, dtype=np.float32), (r, 1))
        
        test_eof(f)
        
    with open(os.path.join(embs_dir, "Z.emb"), 'rb') as f:
        r = struct.unpack('i', f.read(4))[0]
        c = struct.unpack('i', f.read(4))[0]
        Z_embs = np.reshape(np.fromfile(f, dtype=np.float32), (r, c))
        
        test_eof(f)
    
    with open(os.path.join(embs_dir, "Z.bias"), 'rb') as f:
        r = struct.unpack('i', f.read(4))[0]
        Z_bias = np.reshape(np.fromfile(f, dtype=np.float32), (r, 1))
        
        test_eof(f)
    
    X_embs_bias = np.concatenate([X_embs, X_bias, np.ones_like(X_bias)], axis=1)
    Z_embs_bias = np.concatenate([Z_embs, np.ones_like(Z_bias), Z_bias], axis=1)
    
    if verbose:
        print(f"X_embs_bias.shape={X_embs_bias.shape}, Z_embs_bias.shape={Z_embs_bias.shape}")
    
    if save:
        np.save(os.path.join(embs_dir, 'X_embs_bias.npy'), X_embs_bias)
        np.save(os.path.join(embs_dir, 'Z_embs_bias.npy'), Z_embs_bias)
    else:
        return X_embs_bias, Z_embs_bias
    
def main():
    args = parse_args()
    construct_emb_with_bias(args.emb_dir)
    
if __name__ == '__main__':
    main()