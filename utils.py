import torch
import torchvision.transforms as transforms

def normalize_g_synthesised_tens(tens):
  tens = (tens.clamp(-1, 1) + 1) / 2.0
  return tens

def show_from_raw_g_synthesis(tens, normalized = False):

  # base_img = G.synthesis(w, noise_mode = 'const') ...

  if not normalized:
      tens = normalize_g_synthesised_tens(tens)
  
  plt.imshow(trans_to_pil(tens[0]))
  plt.show()

def save_from_raw_g_synthesis(tens, fname, normalized = False):

  # base_img = G.synthesis(w, noise_mode = 'const') ...

  if not normalized:
      tens = normalize_g_synthesised_tens(tens)
  
  trans_to_pil(tens[0]).save(fname)

def print_smol_numbers(s):
  """print the string s but not as verbose as it was!"""

  numcnt=0
  s=str(s)

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
  
  print(fname, image.shape)
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

def mem():
    total_memory = torch.cuda.get_device_properties(0).total_memory
    reserved_memory = torch.cuda.memory_reserved(0) 
    allocated_memory = torch.cuda.memory_allocated(0)
    if debug: return str(allocated_memory / total_memory) + "\n"
    else: return ""

def estimate_memory():
  print(get_memory(), end="")

# def show_grid(images, fname=""):
#   rows = len(images)
#   cols = len(images[0])

#   big_tens = torch.zeros(rows * cols, 3, 1024, 1024)

#   for i in range(rows * cols):
#     cur_im = images[i % rows][i // rows]

#     if str(type(cur_im))[:5] == "torch":
#       big_tens[i] = cur_im
#       print("fake: ")
#       print(big_tens)

#     elif type(cur_im) == type(""):
#       big_tens[i] = trans_to_tensor(Image.open(cur_im))
#       print("proper: ")
#       print(big_tens[i])

#     else: # uh, maybe a PIL image anyways?
#       big_tens[i] = trans_to_tensor(cur_im)

#   grid = torchvision.utils.make_grid(big_tens, nrow=rows)
#   grid_img = trans_to_pil(grid)

#   if len(fname) == 0:
#     plt.imshow(grid_img)
#     plt.show()

#   else:
#     print("hello")
#     grid_img.save(fname)

def show_grid(images, fname="", down_to_256 = None):
  rows = len(images)
  cols = len(images[0])

  big_tens = torch.zeros(rows * cols, 3, 1024, 1024)

  for i in range(rows * cols):
    cur_im = images[i % rows][i // rows]
    if down_to_256: cur_state = down_to_256[i % rows][i // rows]

    cur_tens = None

    if "torch" in str(type(cur_im)):
        cur_tens = cur_im

    elif type(cur_im) == type(""):
        cur_tens = trans_to_tensor(Image.open(cur_im))

    else: # uh, maybe a PIL image anyways?
        cur_tens = trans_to_tensor(cur_im)

    if len(cur_tens.shape) < 4:
        cur_tens = cur_tens.unsqueeze(0)
    
    print(cur_tens.shape)
    if down_to_256 and cur_state:
        cur_tens = torch.nn.functional.interpolate(cur_tens, scale_factor = 0.25)
        # cur_tens = downz(cur_tens)
    print(cur_tens.shape)

    if cur_tens.shape[2] != 1024:
        ups = torch.nn.Upsample(scale_factor = 1024 // cur_tens.shape[2])
        cur_tens = ups(cur_tens)

    big_tens[i] = cur_tens

  grid = torchvision.utils.make_grid(big_tens, nrow=rows)
  grid_img = trans_to_pil(grid)

  if len(fname) == 0:
    plt.imshow(grid_img)
    plt.show()

  else:
    print("hello")
    grid_img.save(fname)

def loss_plot(xs, yss, labels=[]):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    for ys in yss:
        if type(ys[0]) != type([]):
            ys = [ys]
            break

    for i, ys in enumerate(yss):
        ax1.scatter(xs, ys, label = (labels[i] if labels else None))

    if labels: plt.legend()
    plt.show()
    return

def npify(im):
    im = im[0].permute(1, 2, 0).cpu().numpy().squeeze()
    return im

def get_pm1_ssim(tens1, tens2): ## from 'raw g synthesis' ie [-1, 1] and NCHW
    print("showin")
    show_from_raw_g_synthesis(tens1)
    show_from_raw_g_synthesis(tens2)
    return ssim(npify(tens1), npify(tens2), data_range = 2, multichannel=True)

def get_pm1_lpips(tens1, tens2):
    return loss_fn_alex(tens1, tens2).item()

def fpath_to_pm1(fpath, dim=1024):
    pm1 = trans_to_tensor(Image.open(fpath)).to("cuda").reshape(1,3,dim,dim) ## .reshape(1, 3, 1024, 1024) # seems to be in range [0, 1]
    pm1 *= 2
    return pm1 - 1.0

# with open(f'brgmjson/image{i}.json', 'w') as f:
#     json.dump(brgm.loss_log, f)
# with open("brgmjson/image0full.json", "r") as f:
#     my_dict = json.load(f)

trans_to_tensor = transforms.ToTensor()