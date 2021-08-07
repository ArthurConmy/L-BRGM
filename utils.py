from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
trans = transforms.ToPILImage()
trans1 = transforms.ToTensor()

def normalize_g_synthesised_tens(tens):
  tens = (tens.clamp(-1, 1) + 1) / 2.0
  return tens

def show_from_raw_g_synthesis(tens, normalized = False):

  # base_img = G.synthesis(w, noise_mode = 'const') ...

  if not normalized:
      tens = normalize_g_synthesised_tens(tens)
  
  plt.imshow(trans(tens[0]))
  plt.show()

def save_from_raw_g_synthesis(tens, fname, normalized = False):

  # base_img = G.synthesis(w, noise_mode = 'const') ...

  if not normalized:
      tens = normalize_g_synthesised_tens(tens)
  
  trans(tens[0]).save(fname)

  # plt.imsave(fname, trans(tens[0]))
  # plt.show()

def print_smol_numbers(s):
  """print the string s but not as verbose as it was!"""

  numcnt=0

  for i in range(len(s)):
    if ord(s[i]) < ord('0') or ord(s[i]) > ord('9'):
      numcnt=0
    else:
      numcnt+=1
    if numcnt<3: print(s[i],end='')

  print()

def save_image_show(image, target_res=None):
  ''' image = CHW (no batch dimension anymore)'''
  # print('image.shape', image.shape)
  chan = image.shape[0]
  image = image.permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu().numpy().squeeze()
  # print('image.shape', image.shape)
  if chan == 3:
    pilimg = PIL.Image.fromarray(image, 'RGB')
  else:
    # assume grayscale
    pilimg = PIL.Image.fromarray(image, 'L')

  if target_res is not None:
    pilimg = pilimg.resize(target_res, PIL.Image.NEAREST)

  plt.imshow(pilimg)
  plt.show()

def save_image(image, fname, target_res=None):
  ''' image = CHW (no batch dimension anymore)'''
  # print('image.shape', image.shape)
  chan = image.shape[0]
  image = image.permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu().numpy().squeeze()
  # print('image.shape', image.shape)
  if chan == 3:
    pilimg = PIL.Image.fromarray(image, 'RGB')
  else:
    # assume grayscale
    pilimg = PIL.Image.fromarray(image, 'L')

  if target_res is not None:
    pilimg = pilimg.resize(target_res, PIL.Image.NEAREST)

  pilimg.save(fname)
  # plt.imsave(fname, pilimg)
  # plt.show()

def percentage_diff(tens1, tens2, suppress_smol = False):
  t1 = tens1.detach().clone()
  t2 = tens2.detach().clone()

  if torch.linalg.norm(t1) < 1e-8:
    if not suppress_smol:
      print('WARN: t1 close to 0')

  if torch.linalg.norm(t2) < 1e-8:
    if not suppress_smol:
      print('WARN: t2 close to 0')

  return "{:.0%}".format((torch.linalg.norm(t1 - t2) / torch.linalg.norm(t1)).item())