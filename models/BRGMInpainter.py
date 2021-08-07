from models.FFHQInpainter import FFHQInpainter

class BRGMInpainter(FFHQInpainter):
  def __init__(self, ground_truth, verbose = True, im_verbose = True):
    super().__init__(ground_truth, verbose, im_verbose)
    
    self.lambda_pix = 0.001
    self.lambda_perc = 10000000
    self.lambda_w = 100
    self.lambda_c = 0.1

  def forward(self, show_image = False):
    synth_images = self.G.synthesis(self.w, noise_mode='const')  # G(w)
    if show_image and self.im_verbose: show_from_raw_g_synthesis(synth_images, False)
    synth_images = (synth_images + 1) * (256 / 2)
    synth_images_down = self.corrupt(synth_images)
    
    # print(synth_images_down)
    # print(self.target)

    # adding L2 loss in pixel space
    pixelwise_loss = self.lambda_pix * (synth_images_down - self.target).square().mean()

    loss = 0.0
    loss += pixelwise_loss

    # perceptual loss
    synth_features = getVggFeatures(synth_images_down, 3, self.vgg16)
    perceptual_loss = self.lambda_perc * (self.target_features - synth_features).square().mean()
    loss += perceptual_loss

    # adding prior on w ~ N(mu, sigma) as extra loss term
    w_loss = self.lambda_w * (
          self.w / self.w_std_scalar - self.w_avg / self.w_std_scalar).square().mean()  # will broadcast w_avg: [1, 1, 512] to ws: [1, L, 512]
    loss += w_loss

    # adding cosine distance loss
    cosine_loss = self.lambda_c * cosine_distance(self.w)
    loss += cosine_loss

    self.loss_log.append({"total" : loss.item(),
                          "pixelwise" : pixelwise_loss.item(),
                          "perceptual" : perceptual_loss.item(),
                          "w" : w_loss.item(),
                          "cosine" : cosine_loss.item()})
    if self.step_no % self.lossprint_interval == self.lossprint_interval - 1 and self.verbose: print_smol_numbers(str(self.loss_log[-1]))
    return loss

  def initialise_corrupt(self):
    self.mask = ForwardFillMask(self.device)
    self.mask.mask = torch.load("halfmask.pt", map_location=self.device)

  def corrupt(self, tens):
    return self.mask(tens)
    # return tens