# see the image file in the google drive folder to check that this implementation is doing the right thing

from abc import ABC

class FFHQInpainter(torch.nn.Module, ABC):
  def __init__(self, 
               ground_truth,
               verbose = True,
               im_verbose = True):
    
    """
    Parameters:
    -mask is the mask of booleans describing (True) if pixels are blanked out, or not (False)
    -ground_truth is the file path to the 1024x1024 image to be inpainted
    """

    super().__init__()
    self.initialise_logging(verbose, im_verbose)
    self.initialise_hyperparams()

    self.initialise_cuda()
    self.initialise_generator()
    self.initialise_vgg()

    self.initialise_wavg() # needed to set the parameter below
    self.w = torch.nn.Parameter(torch.tensor(self.w_avg.repeat([1, self.G.mapping.num_ws, 1]), dtype=torch.float32, device=self.device, requires_grad=True))    
    self.initialise_corrupt()
    self.test_initial_w()
    
    self.initialise_ground_truth(ground_truth)
    self.test_ground_truth()

    self.initialise_convenience(1)

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
    self.learning_rate = 0.1
    self.beta1 = 0.9
    self.beta2 = 0.9
  def initialise_generator(self):
    network_pkl = "ffhq.pkl" # take this from the BRGM repo/notebook
    if self.verbose: print('Loading networks from "%s"...' % network_pkl)

    time_now = perf_counter()
    with dnnlib.util.open_url(network_pkl) as fp:
      G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(self.device)  # type: ignore
    self.G = copy.deepcopy(G).eval().requires_grad_(False).to(self.device)  # type: ignore
  def initialise_vgg(self):
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'

    if "vgg16.pt" in os.listdir():
      vgg16 = torch.load("vgg16.pt").eval().to(self.device)
    else:
      with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(self.device)
    self.vgg16 = vgg16

    self.loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
  def initialise_wavg(self):
    if "thewavg.pt" in os.listdir() and "wstdscalar.pt" in os.listdir():  
      self.w_avg = torch.load("thewavg.pt", map_location=self.device).to(self.device)
      self.w_std_scalar = torch.load("wstdscalar.pt", map_location=self.device).to(self.device) # I think so?
    else:
      w_avg_samples = w_avg_samples
      if self.verbose: print(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
      z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
      w_samples = G.mapping(torch.from_numpy(z_samples).to(self.device), None)  # [N, L, C]
      w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
      w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
      w_std = torch.tensor(np.std(w_samples, axis=0, keepdims=True), dtype=torch.float32, device=self.device)
      w_std_scalar = torch.tensor((np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5, dtype=torch.float32,
                                  device=self.device)
      w_avg = torch.tensor(w_avg, dtype=torch.float32, device=self.device)  # [1, 1, C]
      torch.save(w_avg, "thewavg.pt")
      torch.save(w_std_scalar, "wstdscalar.pt")
  def initialise_ground_truth(self, ground_truth):
    self.ground_truth = trans1(Image.open(ground_truth)).to(self.device).reshape(1, 3, 1024, 1024) # seems to be in range [0, 1]
    self.ground_truth *= 256
    self.target = self.corrupt(self.ground_truth)
    self.target = self.target.to(self.device).to(torch.float32)
    self.target_features = getVggFeatures(self.target, self.G.img_channels, self.vgg16
    def initialise_corrupt(self): ## initialise the corruption
    pass
  def initialise_convenience(self, lossprint_interval):
    self.lossprint_interval = lossprint_interval

  def test_initial_w(self):
    synth_images = self.G.synthesis(self.w, noise_mode='const')  # G(w)

    if self.im_verbose:
      print("This should be the StyleGAN average face!")
      show_from_raw_g_synthesis(synth_images)
      print("Shown.")

      print("This should be the corrupted face!")
      show_from_raw_g_synthesis(self.corrupt(synth_images))
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
  def forward(self, show_image = False):
    pass
  def get_ground_truth_pm1(self):
    pm1 = self.ground_truth
    pm1 *= (256 / 2)

  def get_current_reconstruction_pm1(self):
    return self.G.synthesis(self.w.detach().clone(), noise_mode='const').detach().clone()
  def get_current_reconstruction(self):
    synth_images = self.get_current_reconstruction_pm1()
    synth_images2 = (synth_images.detach().clone() + 1) * (255 / 2)
    return synth_images2
  def show_merged(self): # if we are doing inpainting show something merged
    synth_images = self.G.synthesis(self.w.detach().clone(), noise_mode='const')  # G(w)
    synth_images2 = (synth_images.detach().clone().to(self.device) + 1) * (255 / 2)
    merged = torch.where(self.mask.mask, synth_images2[0, :, :, :],
                              self.target)[0, :, :, :]
    save_image_show(merged)
  def save_merged(self, fname):
    synth_images = self.G.synthesis(self.w.detach().clone(), noise_mode='const')  # G(w)
    synth_images2 = (synth_images.detach().clone().to(self.device) + 1) * (255 / 2)
    merged = torch.where(self.mask.mask, synth_images2[0, :, :, :],
                              self.target)[0, :, :, :]
    save_image(merged, fname)
  def save_generated(self, fname):
    synth_images = self.G.synthesis(self.w, noise_mode='const')  # G(w)
    save_from_raw_g_synthesis(synth_images, fname, False)
  def get_lpips(self):
    recon = self.get_current_reconstruction_pm1().detach().clone().to('cpu')
    truly = self.ground_truth.detach().clone().to('cpu')
    truly /= (255 / 2)
    truly -= 1.0
    print('yeet')
    show_from_raw_g_synthesis(recon)
    show_from_raw_g_synthesis(truly)
    print(self.loss_fn_alex(recon, truly))
  def train_model(self, timeout = 1e9, max_steps = 1e9, save_the_merged = False, show_image = False):

    if self.im_verbose: show_image = True ## ehh this is not optimal

    opt = torch.optim.Adam(self.parameters(), betas=(self.beta1, self.beta2),
                                      lr = self.learning_rate)

    self.loss_log = []
    self.true_loss_log = []

    start_time = perf_counter()
    self.step_no = 0

    while perf_counter() - start_time < timeout and self.step_no < max_steps:
      opt.zero_grad()
      loss = self.forward(show_image = show_image)
      loss.backward()
      opt.step()
      
      self.step_no += 1 

    if save_the_merged: self.save_merged(f"testing_{ perf_counter() }.png")  