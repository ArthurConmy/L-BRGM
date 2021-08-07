from .models.BRGMInpainter import BRGMInpainter

brgm = BRGMInpainter("datasets/ffhq/1.png", verbose = False, im_verbose = True) ## .to('cpu')
# print(brgm.device)
# brgm.get_lpips()
# input()

brgm.beta1 = 0.9
brgm.beta2 = 0.999
brgm.train_model(max_steps = 20)
loss_log = [entrr["total"] for entrr in brgm.loss_log]
print("Percentage DIFF from expected BRGM losses:", percentage_diff(torch.FloatTensor(loss_log), torch.load("losslog.pt")))
print('FIX THE ISSUE WITH LPIPS NEEDING THINGS TO BE ON SAME DEVICE')

## yikes it takes 2 mins 43 to fix the issues with this shit 