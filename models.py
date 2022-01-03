import torch
from abc import ABC, abstractmethod
from utils import *
from time import perf_counter, ctime
import dnnlib
import legacy
import copy
import os
import numpy as np
from bayesmap_recon import constructForwardModel, getVggFeatures, cosine_distance
import PIL.Image as Image
import torch.nn.functional as F
import lpips
from skimage.metrics import structural_similarity as ssim

class FFHQInpainter(torch.nn.Module, ABC):
  # see the image file in the google drive folder to check that this implementation is doing the right thing
  # ^?

  def __init__(self, ground_truth, verbose = True, im_verbose = True, out_dir = "", hollow=False, trial_no = -1, indx = 0):
    
    """Parameters:
    -mask is the mask of booleans describing (True) if pixels are blanked out, or not (False)
    -ground_truth is the file path to the 1024x1024 image to be inpainted"""

    self.fname = ground_truth
    self.indx = indx
    
    self.out_dir = out_dir
    if len(self.out_dir) != 0 and self.out_dir[-1] != "/": self.out_dir += "/"
    self.min_lpips = 100.0

    self.loss_log = []
    self.true_loss_log = []
    self.hollow = hollow
    if trial_no != -1: self.trial_no = trial_no

    if hollow: return

    # # mem()
    super().__init__()
    # # mem()
    self.initialise_logging(verbose, im_verbose)
    # # mem()
    self.initialise_hyperparams()
    # # mem()

    self.initialise_cuda()
    self.initialise_generator()
    self.initialise_vgg_from_scratch() # TODO fix
    # # mem()

    self.initialise_wavg() # needed to set the parameter below
    # # mem()
    # self.w = torch.nn.Parameter(torch.tensor(self.w_avg.repeat([1, self.G.mapping.num_ws, 1]), dtype=torch.float32, device=self.device, requires_grad=True))    

    if False:
        self.old_z_init()
        self.test_initial_w()

    # # mem()
    self.initialise_corrupt()
    # # mem()
    
    self.initialise_ground_truth(ground_truth)
    # # mem()
    # self.test_initial_w()
    # # mem()
    self.test_ground_truth()
    # # mem()

    self.initialise_convenience(1)
    # mem()
    self.initialise_metrics()
    # mem()
    self.best_lpips = {}

  def old_z_init():
    self.w = torch.nn.Parameter(torch.tensor(self.w_avg.repeat([1, self.G.mapping.num_ws, 1]), dtype=torch.float32, device=self.device, requires_grad=True))    

  def initialise_cuda(self):
    if torch.cuda.is_available():
      if self.verbose: print('Working on GPU, good')
      self.device = torch.device('cuda')
    else:
      if self.verbose: print('Warning, working on CPU')
      self.device = torch.device('cpu')
  def initialise_logging(self, verbose, im_verbose):
    self.verbose = verbose
    self.im_verbose = im_verbose
  def initialise_hyperparams(self):
    self.z_lr = 0.01
    self.w_lr = 0.01
  def initialise_generator(self):
    network_pkl = "ffhq.pkl" # take this from the BRGM repo/notebook
    if self.verbose: print('Loading networks from "%s"...' % network_pkl)
    time_now = perf_counter()
    with dnnlib.util.open_url(network_pkl) as fp:
      G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(self.device)  # type: ignore
    self.G = copy.deepcopy(G).eval().requires_grad_(False).to(self.device)  # type: ignore 
  def initialise_vgg_from_scratch(self):
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    if "vgg16.pt" in os.listdir():
      vgg16 = torch.load("vgg16.pt").eval().to(self.device)
    else:
      with dnnlib.util.open_url(url) as f:
        self.vgg16 = copy.deepcopy(torch.jit.load(f).eval().to(self.device))
  def initialise_vgg_from_global(self):
    # mem()
    self.vgg16 = vgg16_global
    self.vgg16.zero_grad()
    # mem()
  def initialise_wavg(self):
    if "thewavg.pt" in os.listdir() and "wstdscalar.pt" in os.listdir():  
      self.w_avg = torch.load("thewavg.pt", map_location=self.device).to(self.device)
      self.w_std_scalar = torch.load("wstdscalar.pt", map_location=self.device).to(self.device) # I think so?

    #   self.z = 
    else:
      w_avg_samples = 100
      if self.verbose: print(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
      z_samples = np.random.RandomState(123).randn(w_avg_samples, self.G.z_dim)
      w_samples = self.G.mapping(torch.from_numpy(z_samples).to(self.device), None)  # [N, L, C]
      w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
      w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
      w_std = torch.tensor(np.std(w_samples, axis=0, keepdims=True), dtype=torch.float32, device=self.device)
      w_std_scalar = torch.tensor((np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5, dtype=torch.float32,
                                  device=self.device)
      w_avg = torch.tensor(w_avg, dtype=torch.float32, device=self.device)  # [1, 1, C]
      torch.save(w_avg, "thewavg.pt")
      torch.save(w_std_scalar, "wstdscalar.pt")
  def initialise_ground_truth(self, ground_truth):
    self.ground_truth = trans_to_tensor(Image.open(ground_truth)).to(self.device).reshape(1, 3, 1024, 1024) # seems to be in range [0, 1]
    self.ground_truth *= 255

    self.ground_truth_pm1 = (self.ground_truth / (255 / 2)) - 1.0
    self.ground_truth_pm1_256 = F.interpolate(self.ground_truth_pm1, scale_factor=0.25)

    self.target = self.corrupt(self.ground_truth)
    self.target = self.target.to(self.device).to(torch.float32)
    self.target_features = getVggFeatures(self.target, self.G.img_channels, self.vgg16)  
  def initialise_corrupt(self): ## initialise the corruption
    pass
  def initialise_convenience(self, lossprint_interval):
    self.lossprint_interval = lossprint_interval
  def initialise_metrics(self):
    self.loss_fn_alex = lpips.LPIPS(net='alex').to(self.device) # best forward scores

  def test_initial_w(self):
    synth_images = self.G.synthesis(self.w, noise_mode='const')  # G(w)

    if self.im_verbose:
      print("This should be the StyleGAN average face!")
      show_from_raw_g_synthesis(synth_images)
      # print("Shown.")

      print("This should be the corrupted face!")
      save_image_show(self.corrupt(self.ground_truth)[0, :, :, :])
      
      # corrupted_im = self.target.detach().clone() / (255 / 2)
      # corrupted_im -= 1.0
      # plt.imshow(trans_to_pil(self.target[0]))
      # plt.show()
      print("Shown.")
  def test_ground_truth(self):
    if self.im_verbose:
      print("This should be the ground truth:")
      save_image_show(self.ground_truth[0, :, :, :])
      print("Shown.")

      print("This should be the corrupted face!")
      save_image_show(self.corrupt(self.ground_truth)[0, :, :, :])
      print("Shown.")

  def corrupt(self, tens): ## corrupt the tensor tens
    return tens

  @abstractmethod
  def forward(self):
    pass
  def get_current_reconstruction_pm1(self):
    return self.G.synthesis(self.w.detach().clone(), noise_mode='const').detach().clone()
  def get_current_reconstruction_pm1_256(self):
    cur = self.get_current_reconstruction_pm1()
    cur = F.interpolate(cur, scale_factor=0.25)
    return cur
  def get_current_merged_pm1(self):
    synth_images = self.G.synthesis(self.w.detach().clone(), noise_mode='const')  # G(w)
    synth_images2 = (synth_images.detach().clone().to(self.device) + 1) * (255 / 2)
    merged = torch.where(self.mask.mask, synth_images2[0, :, :, :],
                              self.target)
    merged /= (255 / 2)
    merged -= 1.0
    return merged
  def get_current_merged_pm1_256(self):
      cur = self.get_current_merged_pm1()
      return F.interpolate(cur, scale_factor=0.25)
  def get_current_reconstruction(self):
    synth_images = self.get_current_reconstruction_pm1()
    synth_images2 = (synth_images.detach().clone() + 1) * (255 / 2)
    return synth_images2
  def save_corrupted_gt(self, fname):
    # merged = torch.where(self.mask.mask, synth_images2[0, :, :, :],
                            #   self.target)[0, :, :, :]
    print("Trying to save")
    save_image(self.target[0, :, :, :], self.out_dir + f"{fname}")
    print("Tried to save!")
  def show_merged(self): # if we are doing inpainting show something merged
    synth_images = self.G.synthesis(self.w.detach().clone(), noise_mode='const')  # G(w)
    synth_images2 = (synth_images.detach().clone().to(self.device) + 1) * (255 / 2)
    merged = torch.where(self.mask.mask, synth_images2[0, :, :, :],
                              self.target)[0, :, :, :]
    save_image_show(merged)
  def show_generated(self, corrupted=False):
    synth_images = self.G.synthesis(self.w.detach().clone(), noise_mode='const')
    if corrupted: synth_images = self.corrupt(synth_images)
    show_from_raw_g_synthesis(synth_images)
  def save_merged(self, fname):
    synth_images = self.G.synthesis(self.w.detach().clone(), noise_mode='const')  # G(w)
    synth_images2 = (synth_images.detach().clone().to(self.device) + 1) * (255 / 2)
    merged = torch.where(self.mask.mask, synth_images2[0, :, :, :],
                              self.target)[0, :, :, :]
    save_image(merged, self.out_dir + fname)
  def save_the_image(self, fname):
    synth_images = self.G.synthesis(self.w, noise_mode='const')  # G(w)
    save_from_raw_g_synthesis(synth_images, self.out_dir + fname, False)
  def show_merged(self):
    synth_images = self.G.synthesis(self.w.detach().clone(), noise_mode='const')  # G(w)
    synth_images2 = (synth_images.detach().clone().to(self.device) + 1) * (255 / 2)
    merged = torch.where(self.mask.mask, synth_images2[0, :, :, :],
                              self.target)[0, :, :, :]
    save_image_show(merged)

  def get_lpips(self):
    recon = self.get_current_reconstruction_pm1().detach().clone()
    truly = self.ground_truth.detach().clone()
    truly /= (255 / 2)
    truly -= 1.0

    return self.loss_fn_alex(recon, truly).item()

  def get_ssim(self):
    recon = self.get_current_reconstruction_pm1().detach().clone()
    truly = self.ground_truth.detach().clone()
    truly /= (255 / 2)
    truly -= 1.0     

    return ssim(npify(recon), npify(truly), data_range = 2, multichannel=True)

  def get_down_lpips_merged(self):
    recon = self.get_current_merged_pm1_256() ## self.get_current_reconstruction_pm1().detach().clone()
    truly = self.ground_truth_pm1_256 ## self.ground_truth.detach().clone(

    print(recon.shape, truly.shape, "are the shapes")
    return self.loss_fn_alex(recon, truly).item()

  def get_down_ssim_merged(self):
      recon = self.get_current_merged_pm1_256()
      truly = self.ground_truth_pm1_256
      return get_pm1_ssim(recon, truly)

  def get_lpips_sr(self):
      recon = self.get_current_reconstruction_pm1_256()
      truly = self.ground_truth_pm1_256
      return loss_fn_alex(recon, truly).item()

  def get_ssim_sr(self):
      recon = self.get_current_reconstruction_pm1_256()
      truly = self.ground_truth_pm1_256
    
      

      return get_pm1_ssim(recon, truly)

  def get_comp_lpips(self):
    recon = self.corrupt(self.get_current_reconstruction_pm1().detach().clone())
    truly = self.corrupt(self.ground_truth.detach().clone())
    truly /= (255 / 2)
    truly -= 1.0

    return self.loss_fn_alex(recon, truly).item()

  def get_comp_ssim(self):
    recon = self.corrupt(self.get_current_reconstruction_pm1().detach().clone())
    truly = self.corrupt(self.ground_truth.detach().clone())
    truly /= (255 / 2)
    truly -= 1.0

    # print(otherer.device, truly.device, "devices")

    return ssim(npify(recon), npify(truly), data_range = 2, multichannel=True)

  def get_other_lpipz(self, otherer):
      truly = self.corrupt(self.ground_truth.detach().clone()).detach().clone()
      truly /= (255 / 2)
      truly -= 1.0

      return self.loss_fn_alex(otherer.to(self.device), truly).item()

  def get_other_ssim(self, otherer):
      truly = self.corrupt(self.ground_truth.detach().clone())
      truly /= (255 / 2)
      truly -= 1.0

    #   print(truly.shape, otherer.shape)

      return ssim(npify(otherer), npify(truly), data_range = 2, multichannel=True)

  def train_model(self, timeout = 1e9, max_steps = 1e9, save_the_merged = False):
    self.max_steps=max_steps
    # if len(self.zparams) == 0:
    # opt = torch.optim.Adam([self.z, self.w], betas=(self.beta1, self.beta2), lr = 0.1) ## self.learning_rate)

    # else:

    adam_list = [{"params": [self.w], "lr": self.w_lr}]

    if "z" in self.__dict__:
        adam_list.append({"params": [self.z], "lr": self.z_lr})

    opt = torch.optim.Adam(adam_list)

    # opt_z = torch.optim.Adam([self.z], lr=self.z_lr)
    # opt_w = torch.optim.Adam([self.w], lr=self.w_lr)

    self.loss_log = []
    self.true_loss_log = []

    start_time = perf_counter()
    self.step_no = 0

    while perf_counter() - start_time < timeout and self.step_no < max_steps:
    #   opt_z.zero_grad()
    #   opt_w.zero_grad()
      opt.zero_grad()

      loss = self.forward()

      if self.step_no % self.lossprint_interval == self.lossprint_interval - 1:
        print(self.step_no, 'is the step number, and losses') 
        print_smol_numbers("RAW: " + str(self.loss_log[-1]))
        print_smol_numbers("TRUE: " + str(self.true_loss_log[-1]))
        print(ctime())
        print()

      loss.backward()
    #   print('stepping ... ?')
      opt.step()
    #   opt_w.step()

      self.step_no += 1

    if save_the_merged: self.save_merged(f"testing_{ perf_counter() }.png")
    

class BRGMInpainter(FFHQInpainter):
  def __init__(self, *args, **kwargs):
    print(f"*args: {args}, **kwargs: {kwargs}")
    super().__init__(*args, **kwargs)    
    self.beta1 = 0.9
    self.beta2 = 0.999

    if not self.hollow: 
        self.w = torch.nn.Parameter(torch.tensor(self.w_avg.repeat([1, self.G.mapping.num_ws, 1]), dtype=torch.float32, device=self.device, requires_grad=True))
        print(f"self.wshape {self.w.shape}")

    self.lambda_pix = 0.001
    self.lambda_perc = 10000000
    self.lambda_w = 100
    self.lambda_c = 0.1
    self.learning_rate = 0.1

    self.blambda_pix = 0.001
    self.blambda_perc = 10000000
    self.blambda_w = 500 # note 5x the BRGM value
    self.blambda_c = 0.1

    self.min_lpips = 100.0

    # self.initialise_new_w()
    if not self.hollow:
        self.test_initial_w()

  def initialise_new_w(self):

      min_percep_loss = 100.0

      for trail_z in range(100):
          new_z = torch.from_numpy(np.random.randn(1, self.G.z_dim)).to(self.device)
          new_w = self.G.mapping(new_z, None)
          
          self.w2 = new_w
          print('Trying', self.get_losses()["perceptual"])
          if self.get_losses()["perceptual"] < min_percep_loss:
              min_percep_loss = self.get_losses()["perceptual"]
              self.z = new_z

      self.w = torch.nn.Parameter(self.G.mapping(self.z, None).to(self.device))


  def forward(self):
    synth_images = self.G.synthesis(self.w, noise_mode='const')  # G(w)
    if self.im_verbose: self.show_generated()
    ## show_from_raw_g_synthesis(synth_images, False)
    synth_images = (synth_images + 1) * (255 / 2)
    synth_images_down = self.corrupt(synth_images)
    
    # print(synth_images_down)
    # print(self.target)

    # adding L2 loss in pixel space
    pixelwise_loss = (synth_images_down - self.target).square().mean()

    loss = 0.0
    loss += self.lambda_pix * pixelwise_loss

    # perceptual loss
    synth_features = getVggFeatures(synth_images_down, 3, self.vgg16)
    perceptual_loss = (self.target_features - synth_features).square().mean()
    loss += self.lambda_perc * perceptual_loss

    # adding prior on w ~ N(mu, sigma) as extra loss term
    w_loss = (
          self.w / self.w_std_scalar - self.w_avg / self.w_std_scalar).square().mean()  # will broadcast w_avg: [1, 1, 512] to ws: [1, L, 512]
    loss += self.lambda_w * w_loss

    # adding cosine distance loss
    cosine_loss = cosine_distance(self.w)
    loss += self.lambda_c * cosine_loss

    self.cur_lpips = self.get_lpips()

    self.true_loss_log.append({"total" : loss.item(),
                          "pixelwise" : self.lambda_pix * pixelwise_loss.item(),
                          "perceptual" : self.lambda_perc * perceptual_loss.item(),
                          "w" : self.lambda_w * w_loss.item(),
                          "cosine" : self.lambda_c * cosine_loss.item()})

    self.loss_log.append({"total" : loss.item(),
                          "pixelwise" : pixelwise_loss.item(),
                          "perceptual" : perceptual_loss.item(),
                          "w" : w_loss.item(),
                          "cosine" : cosine_loss.item(),
                          "lpips" : self.cur_lpips})
    
    
    if self.step_no == self.max_steps - 1: self.loss_log[-1]["ssim"] = self.get_ssim()
    
    if self.fname not in self.best_lpips or self.cur_lpips < self.best_lpips[self.fname] or self.step_no == self.max_steps-1:
        self.best_lpips[self.fname] = self.cur_lpips
            # print(self.fname, self.learning_rate, self.lambda_perc, self.lambda_pix, self.lambda_mse, self.lambda_norm)
        self.show_generated()
        self.save_the_image(f"{self.step_no} {ctime()}.png")
        torch.save(self.w.detach().clone(), self.out_dir + f"{self.step_no} {ctime()}.pt")

    return loss

  def initialise_corrupt(self):
    self.downres, _ = constructForwardModel("super-resolution", self.G.img_resolution, self.G.img_channels, None, self.fname, ## I guess ...
                                          1 / 16, 0, self.device)
    
    # self.mask = ForwardFillMask(self.device)
    # self.mask.mask = torch.load("halfmask.pt", map_location=self.device)


  def corrupt(self, tens):
    return self.downres(tens)
    # return self.mask(tens)

  # old things 
    # self.mask = ForwardFillMask(self.device)
    # self.mask.mask = torch.load("halfmask.pt", map_location=self.device)

  # def corrupt(self, tens):
  #   return self.mask(tens)
  #   # return tens

  def get_losses(self): # take some vector w2 and send it through, return the losses

    # self.w2 = self.w.detach().clone().to(self.device)
    synth_images = self.G.synthesis(self.w2, noise_mode='const')  # G(w)
    synth_images = (synth_images + 1) * (255 / 2)
    synth_images_down = self.corrupt(synth_images)
    
    pixelwise_loss = (synth_images_down - self.target).square().mean()
    loss = 0.0
    loss += self.blambda_pix * pixelwise_loss

    synth_features = getVggFeatures(synth_images_down, 3, self.vgg16)
    perceptual_loss = (self.target_features - synth_features).square().mean()
    loss += self.blambda_perc * perceptual_loss

    w_loss = (
          self.w2 / self.w_std_scalar - self.w_avg / self.w_std_scalar).square().mean()
    loss += self.blambda_w * w_loss

    cosine_loss = cosine_distance(self.w2)
    loss += self.blambda_c * cosine_loss

    # self.cur_lpips = self.get_lpips(
    
    return {"total" : loss.item(),
                          "pixelwise" : self.blambda_pix * pixelwise_loss.item(),
                          "perceptual" : self.blambda_perc * perceptual_loss.item(),
                          "w" : self.blambda_w * w_loss.item(),
                          "cosine" : self.blambda_c * cosine_loss.item(),
                        #   "lpips" : self.cur_lpips}
    }

  def train_brgm(self, timeout = 1e9, max_steps = 1e9, save_the_merged = False):
    self.max_steps = max_steps
    opt = torch.optim.Adam([self.w], lr=self.learning_rate)

    self.loss_log = []
    self.true_loss_log = []

    start_time = perf_counter()
    self.step_no = 0

    while perf_counter() - start_time < timeout and self.step_no < max_steps:
      opt.zero_grad()

      loss = self.forward()

      if self.step_no % self.lossprint_interval == self.lossprint_interval - 1:
        print(self.step_no, 'is the step number, and losses') 
        print_smol_numbers("RAW: " + str(self.loss_log[-1]))
        print_smol_numbers("TRUE: " + str(self.true_loss_log[-1]))
        print(ctime())
        print()
        # self.save_the_image(ctime() + ".png")

      loss.backward()
      opt.step()
      self.step_no += 1

    if save_the_merged: self.save_merged(f"testing_{ perf_counter() }.png")
    
class SubhadipInpainter(FFHQInpainter):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.lambda_pix = 0.0002
    self.lambda_perc = 20000000
    self.lambda_mse = 30
    self.lambda_norm = 0.4
    self.z_lr = 0.1
    self.w_lr = 0.1

    self.blambda_pix = 0.001
    self.blambda_perc = 10000000
    self.blambda_w = 500 # note 5x the BRGM value
    self.blambda_c = 0.1

    self.beta1 = 0.96
    self.beta2 = 0.9999

    if not self.hollow:
        self.initialise_new_zw()
        print(f"self.wshape {self.w.shape}")

  def initialise_new_zw(self):

      min_percep_loss = 100.0

      for trail_z in range(100):
          new_z = torch.from_numpy(np.random.randn(1, self.G.z_dim)).to(self.device)
          new_w = self.G.mapping(new_z, None)
          
          self.w2 = new_w
          if self.verbose and trail_z % 10 == 0: print('Trying', self.get_losses()["perceptual"])
          if self.get_losses()["perceptual"] < min_percep_loss:
              min_percep_loss = self.get_losses()["perceptual"]
              self.z = new_z

    #   self.z = torch.from_numpy(np.random.randn(1, self.G.z_dim)).to(self.device)
      self.w = self.G.mapping(self.z, None)
      self.z.requires_grad = True
      self.w.requires_grad = True

  def initialise_z(self):
    self.z = (self.w.detach().clone())[:, 0, :] ## torch.nn.Parameter(torch.from_numpy(np.random.RandomState(123).randn(1, self.G.z_dim)).to(self.device))

  def get_losses(self): # take some vector w2 and send it through, return the losses

    # self.w2 = self.w.detach().clone().to(self.device)
    synth_images = self.G.synthesis(self.w2, noise_mode='const')  # G(w)
    synth_images = (synth_images + 1) * (255 / 2)
    synth_images_down = self.corrupt(synth_images)
    
    pixelwise_loss = (synth_images_down - self.target).square().mean()
    loss = 0.0
    loss += self.blambda_pix * pixelwise_loss

    synth_features = getVggFeatures(synth_images_down, 3, self.vgg16)
    perceptual_loss = (self.target_features - synth_features).square().mean()
    loss += self.blambda_perc * perceptual_loss

    w_loss = (
          self.w2 / self.w_std_scalar - self.w_avg / self.w_std_scalar).square().mean()
    loss += self.blambda_w * w_loss

    cosine_loss = cosine_distance(self.w2)
    loss += self.blambda_c * cosine_loss

    # self.cur_lpips = self.get_lpips()
    
    return {"total" : loss.item(),
                          "pixelwise" : self.blambda_pix * pixelwise_loss.item(),
                          "perceptual" : self.blambda_perc * perceptual_loss.item(),
                          "w" : self.blambda_w * w_loss.item(),
                          "cosine" : self.blambda_c * cosine_loss.item(),
                        #   "lpips" : self.cur_lpips}
    }
    
  def forward(self):

    synth_images = self.G.synthesis(self.w, noise_mode='const')  # G(w)

    synth_images = (synth_images + 1) * (255 / 2)
    synth_images_down = self.corrupt(synth_images)

    # adding L2 loss in pixel space
    pixelwise_loss = (synth_images_down - self.target).square().mean()

    loss = 0.0
    loss += self.lambda_pix * pixelwise_loss

    # perceptual loss
    synth_features = getVggFeatures(synth_images_down, 3, self.vgg16)
    perceptual_loss = (self.target_features - synth_features).square().mean()
    loss += self.lambda_perc * perceptual_loss

    mse_loss = 0.0
    w_samples = self.G.mapping(self.z, None)  # [N, L, C]
    mse_loss += (w_samples - self.w).square().mean() ## MSE <- it is this key line that causes the optimization to go funny !!!
    loss += self.lambda_mse * mse_loss

    norm_loss = 0.0
    norm_loss += self.z.square().mean()
    loss += self.lambda_norm * norm_loss

    if self.im_verbose: 
        print('Z IMAGE:')
        print(w_samples)
        show_from_raw_g_synthesis(self.G.synthesis(w_samples.detach().clone(), noise_mode = 'const'))

        print('W IMAGE')
        show_from_raw_g_synthesis(self.G.synthesis(self.w.detach().clone(), noise_mode = 'const'))
        print('Shown')

    self.true_loss_log.append({"total" : loss.item(),
                          "pixelwise" : self.lambda_pix * pixelwise_loss.item(),
                          "perceptual" : self.lambda_perc * perceptual_loss.item(),
                          "mse" : self.lambda_mse * mse_loss.item(),
                          "norm" : self.lambda_norm * norm_loss.item()})
    
    self.cur_lpips = self.get_lpips()

    self.loss_log.append({"total" : loss.item(),
                      "pixelwise" : pixelwise_loss.item(),
                      "perceptual" : perceptual_loss.item(),
                      "mse" : mse_loss.item(),
                      "norm" : norm_loss.item(),
                      "lpips" : self.cur_lpips})
                    #    "ssim" : self.get_ssim()})

    if self.fname not in self.best_lpips or self.cur_lpips < self.best_lpips[self.fname] or self.step_no == 1999:
        self.best_lpips[self.fname] = self.cur_lpips
        self.loss_log[-1]["ssim"] = self.get_ssim()
        if self.trial_no > 10:
            # print(self.fname, self.learning_rate, self.lambda_perc, self.lambda_pix, self.lambda_mse, self.lambda_norm)
            self.show_generated()
            self.save_the_image(f"{self.step_no} {ctime()}.png")
            torch.save(self.w.detach().clone(), self.out_dir + f"{self.step_no} {ctime()}.pt")

    if self.im_verbose and self.step_no % 10 == 0:
      if 'downres' not in dir(self): self.show_merged()
      else: self.show_generated()

    return loss

#   def initialise_corrupt(self):
#     self.mask = ForwardFillMask(self.device)
#     self.mask.mask = torch.load("halfmask.pt")
#     # generation of other masks
#     # print(self.mask.mask[0, :, 0, 0])
#     # print(self.mask.mask[0, :, 1023, 1023])
#     # for i in range(767):
#     #   for j in range(1024): 
#     #     self.mask.mask[0, :, j, i] = self.mask.mask[0, :, 0, 0]
#     # self.downres = ForwardDownsample(factor=16).to(self.device)
#     # self.downres, _ = constructForwardModel("super-resolution", self.G.img_resolution, self.G.img_channels, None, self.fname, ## I guess ...
#                                       #  1 / 8, 0, self.device)

#   def corrupt(self, tens):
#     # return self.downres(tens)
#     return self.mask(tens)

  def initialise_corrupt(self):
    # self.downres, _ = constructForwardModel("super-resolution", self.G.img_resolution, self.G.img_channels, None, self.fname, ## I guess ...
                                        #   1 / 4, 0, self.device)
    
    self.mask = ForwardFillMask(self.device)
    self.mask.mask = torch.load("halfmask.pt", map_location=self.device) # or halfmask.pt
    
    # smol_mask = torch.BoolTensor(1, 3, 1024, 1024).fill_(False)
    # smol_mask[0,:,200:500,200:500].fill_(True)
    # smol_mask = smol_mask.to(self.device)
    # torch.save(smol_mask, "eyepatch.pt")
    # self.mask.mask = smol_mask.to(self.device)
    # print(smol_mask[0][0][0][0])
    # # print(smol_mask.shape)

  def corrupt(self, tens):
    # return self.downres(tens)
    return self.mask(tens)