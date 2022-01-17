import os
import nacl
from models import BRGM, LBRGM
import torch
import json
import click
import warnings
warnings.filterwarnings('ignore')

MODELS = {"BRGM" : BRGM, "LBRGM" : LBRGM}

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
@click.option('--model', type=click.Choice(['LBRGM', 'BRGM'],), default='LBRGM')
@click.option('--fpaths', default=get_testable_fpaths(), multiple=True, help='Paths to image file.')
@click.option('--outpath', required=True, help='Output directory to save run progress.')
@click.option('--no-steps', default=2000, help='Number of optimization steps.')
@click.option('--reconstruction-type', type=click.Choice(['inpaint', 'superres'],), help='Corruption process: either inpainting or superresolution.')
@click.option('--input-dim', default=64, help='Height and width of input image to have super-resolution applied')
@click.option('--fpath-corrupted', default=True, help='Whether the input image has already had the corruption applied.')
@click.option('--mask', help='Specify path to the mask to be applied. See masks/1024x1024/ directory for masks')
def run(device, fpaths, outpath, no_steps, reconstruction_type, input_dim, fpath_corrupted, model, mask):
  best_lpips = {}
  if not os.path.exists(outpath): os.mkdir(outpath)
  
  if reconstruction_type == 'superres':
    assert 1024 % input_dim == 0, "Input dimension need be a divisor of 1024, the height/width of images we can generate"
    assert input_dim is not None, "Specify an input dimension"

  if reconstruction_type == 'inpaint':
    assert mask is not None, "Specify a mask to apply. See "

  for i, fpath in enumerate(fpaths):
    print('-'*50)
    print(f"Reconstructing image with file path {fpath}, image {i+1} of {len(fpaths)}")
    print('-'*50)

    cur_outdir = f"{outpath}/{i}"
    if not os.path.exists(cur_outdir): 
      os.mkdir(cur_outdir)
    
    model_args = {
      "fname" : fpath, 
      "verbose" : False, 
      "im_verbose" : True, 
      "out_dir" : cur_outdir, 
      "device" : device, 
      "fpath_corrupted" : fpath_corrupted, 
      "reconstruction_type" : reconstruction_type, 
      "input_dim" : input_dim,
      "mask_file" : mask,
    }

    model = MODELS[model](**model_args)
    model.im_verbose = False
    model.lossprint_interval = 250
    model.learning_rate = 0.1
    model.train_model(max_steps=no_steps)

    # with open(cur_outdir + f'image{i}.json', 'w') as f:
    #   brgmll = json.dump(brgm.loss_log, f)

    # with open(cur_outdir + f'image{i}full.json', 'w') as f:
    #   brgmllf = json.dump(brgm.true_loss_log, f)
    # continue

if __name__ == "__main__":
  run()