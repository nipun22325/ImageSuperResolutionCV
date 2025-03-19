import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips

model_path = '/content/ESRGAN/models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')

test_img_folder = '/content/ESRGAN/LR/LR/*'
hr_img_folder = '/content/ESRGAN/LR/HR/'  # Path to high-resolution ground truth images

# Initialize LPIPS model
loss_fn_alex = lpips.LPIPS(net='alex').to(device)

# Initialize model
model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))

metrics = {'psnr': [], 'ssim': [], 'lpips': []}

idx = 0
for path in glob.glob(test_img_folder):
    idx += 1
    base = osp.splitext(osp.basename(path))[0]
    print(idx, base)
    
    # Handle "x2" suffix in LR image names
    hr_base = base.replace('x2', '') if 'x2' in base else base
    
    # Read LR image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    # Generate SR image
    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output_uint8 = (output * 255.0).round().astype(np.uint8)
    
    # Save output image
    cv2.imwrite('/content/ESRGAN/results/{:s}_rlt.png'.format(base), output_uint8)
    
    # Read corresponding HR image (with corrected base name)
    hr_path = osp.join(hr_img_folder, '{:s}.png'.format(hr_base))  # Adjust file extension if needed
    
    # Check for other common extensions if .png not found
    if not osp.exists(hr_path):
        for ext in ['.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
            alt_path = osp.join(hr_img_folder, '{:s}{:s}'.format(hr_base, ext))
            if osp.exists(alt_path):
                hr_path = alt_path
                break
    
    if osp.exists(hr_path):
        # Load HR image
        img_HR = cv2.imread(hr_path, cv2.IMREAD_COLOR)
        
        # Make sure HR image has the same dimensions as the output
        if img_HR.shape != output_uint8.shape:
            img_HR = cv2.resize(img_HR, (output_uint8.shape[1], output_uint8.shape[0]))
        
        # Calculate PSNR
        psnr_value = psnr(img_HR, output_uint8)
        metrics['psnr'].append(psnr_value)
        
        # Calculate SSIM (convert to grayscale for SSIM)
        img_HR_gray = cv2.cvtColor(img_HR, cv2.COLOR_BGR2GRAY)
        output_gray = cv2.cvtColor(output_uint8, cv2.COLOR_BGR2GRAY)
        ssim_value = ssim(img_HR_gray, output_gray)
        metrics['ssim'].append(ssim_value)
        
        # Calculate LPIPS
        img_HR_tensor = torch.from_numpy(np.transpose(img_HR[:, :, [2, 1, 0]], (2, 0, 1))).float() / 255.0
        img_HR_tensor = img_HR_tensor.unsqueeze(0).to(device)
        
        output_tensor = torch.from_numpy(np.transpose(output_uint8[:, :, [2, 1, 0]], (2, 0, 1))).float() / 255.0
        output_tensor = output_tensor.unsqueeze(0).to(device)
        
        lpips_value = loss_fn_alex(output_tensor, img_HR_tensor).item()
        metrics['lpips'].append(lpips_value)
        
        print(f"Image: {base}, PSNR: {psnr_value:.4f}, SSIM: {ssim_value:.4f}, LPIPS: {lpips_value:.4f}")
    else:
        print(f"Warning: No matching HR image found for {base} (tried {hr_path})")

# Calculate and print averages
if metrics['psnr']:
    avg_psnr = sum(metrics['psnr']) / len(metrics['psnr'])
    avg_ssim = sum(metrics['ssim']) / len(metrics['ssim'])
    avg_lpips = sum(metrics['lpips']) / len(metrics['lpips'])
    
    print(f"\nAverage PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average LPIPS: {avg_lpips:.4f}")
else:
    print("No HR images found for comparison.")