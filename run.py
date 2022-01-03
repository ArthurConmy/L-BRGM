import os
from models import BRGMInpainter
import torch

print(f"Availability of GPU: {torch.cuda.is_available()}")

with open("testables.txt", "r") as f:
  lines = f.readlines()
  lines = [line[:-1] for line in lines]
  lines = [f"images1024x1024/{line[:2]}000/{line}.png" for line in lines]
  testables = lines[:-1]

best_lpips = {}
device='cpu'

for i, testable in enumerate(testables):
  print('-'*50)
  print(i)
  print('-'*50)

  no_steps = 5

  out_dir = f"mynew256to1024/again{i}/"
  im_string = f"256lows/{i}as1024.png"

  if not os.path.exists(out_dir): os.mkdir(out_dir)
  brgm = BRGMInpainter(im_string, verbose=False, im_verbose=False, out_dir = out_dir)
  brgm.im_verbose = False
  brgm.lossprint_interval = 250
  brgm.learning_rate = 0.1
  brgm.train_brgm(max_steps=no_steps)

  with open(out_dir + f'image{i}.json', 'w') as f:
    brgmll = json.dump(brgm.loss_log, f)

  with open(out_dir + f'image{i}full.json', 'w') as f:
    brgmllf = json.dump(brgm.true_loss_log, f)
  continue