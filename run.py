import os
import nacl
from models import BRGM, LBRGM
import torch
import json
import click
import warnings
warnings.filterwarnings('ignore')

def get_testable_fpaths():
  with open("testables.txt", "r") as f:
    lines = f.readlines()
    lines = [line[:-1] for line in lines]
    lines = [f"images1024x1024/{line[:2]}000/{line}.png" for line in lines]
    testables = lines[:-1]
  fpaths = []
  for i in range(len(testables)):
    im_string = f"256lows/{i}as1024.png"
    fpaths.append(im_string)
  return fpaths

@click.command()
@click.option('--device', default=None, help='Device to train on.')
@click.option('--fpaths', default=get_testable_fpaths(), multiple=True, help='Paths to image file.')
@click.option('--outpath', required=True, help='Output directory to save run progress.')
@click.option('--no_steps', default=2000, help='Number of optimization steps.')
@click.option('--reconstruction-type', type=click.Choice(['inpaint', 'superres'],), help='Corruption process: either inpainting or superresolution.')
@click.option('--fpath-corrupted', default=True, help='Whether the input image has already had the corruption applied.')
def run(device, fpaths, outpath, no_steps, reconstruction_type, fpath_corrupted):
  best_lpips = {}

  if not os.path.exists(outpath): os.mkdir(outpath)

  for i, fpath in enumerate(fpaths):
    print('-'*50)
    print(f"Reconstructing image with file path {fpath}, image {i+1} of {len(fpaths)}")
    print('-'*50)

    no_steps = 2000

    cur_outdir = f"{outpath}/{i}"
    if not os.path.exists(cur_outdir): 
      os.mkdir(cur_outdir)

    brgm = BRGM(fpath, verbose=False, im_verbose=True, out_dir = cur_outdir, device=device, fpath_corrupted=fpath_corrupted)
    brgm.im_verbose = False
    brgm.lossprint_interval = 250
    brgm.learning_rate = 0.1
    brgm.train_brgm(max_steps=no_steps)

    # with open(cur_outdir + f'image{i}.json', 'w') as f:
    #   brgmll = json.dump(brgm.loss_log, f)

    # with open(cur_outdir + f'image{i}full.json', 'w') as f:
    #   brgmllf = json.dump(brgm.true_loss_log, f)
    # continue

if __name__ == "__main__":
  run()