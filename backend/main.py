import io
import os
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import cv2
import torch
from typing import Optional
import uvicorn
import logging
from pathlib import Path
import base64

# Import DRCT model architecture
from DRCT.drct.archs.DRCT_arch import DRCT

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="EnhanceIt API", description="API for enhancing images using DRCT model")

# Configure CORS to allow requests from your React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories for storing uploaded and enhanced images
UPLOAD_DIR = Path("uploads")
ENHANCED_DIR = Path("enhanced")
UPLOAD_DIR.mkdir(exist_ok=True)
ENHANCED_DIR.mkdir(exist_ok=True)

# Path to the DRCT model
MODEL_PATH = "models/net_g_latest.pth"  # Update this path to your model location

class ImageEnhancementModel:
    def __init__(self):
        logger.info("Initializing DRCT image enhancement model...")
        
        # Check if CUDA is available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # Initialize the DRCT model
        self.model = DRCT(
            upscale=4, in_chans=3, img_size=64, window_size=16, compress_ratio=3,
            squeeze_factor=30, conv_scale=0.01, overlap_ratio=0.5, img_range=1.,
            depths=[6]*12, embed_dim=180, num_heads=[6]*12, gc=32, mlp_ratio=2,
            upsampler='pixelshuffle', resi_connection='1conv'
        )

        # Load the model weights
        try:
            # When loading for inference in production, we'll load from the saved path
            logger.info(f"Loading model from {MODEL_PATH}")
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device)['params'], strict=True)
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            logger.info("Initializing with random weights (for testing only)")
            # In case the model file is not available, we'll continue without loading weights
            # This is only for testing - in production you must have the model weights

        self.model.eval()
        self.model = self.model.to(self.device)
        self.window_size = 16
        self.scale = 4  # 4x upscaling
        
    def test(self, img_lq):
        """
        Test the model with tiling support for handling large images
        
        Args:
            img_lq: Low quality image tensor
            
        Returns:
            Enhanced image tensor
        """
        # Define tile parameters
        tile = 256  # Tile size, should be divisible by window_size (16)
        tile_overlap = 32  # Overlap size between tiles
        
        # If the image is small enough, process it directly
        if img_lq.size(2) <= tile and img_lq.size(3) <= tile:
            output = self.model(img_lq)
        else:
            # Process with tiling
            b, c, h, w = img_lq.size()
            
            # Make sure tile size is divisible by window_size
            assert tile % self.window_size == 0
            stride = tile - tile_overlap
            sf = self.scale

            h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
            w_idx_list = list(range(0, w-tile, stride)) + [w-tile]

            E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                    out_patch = self.model(in_patch)
                    out_patch_mask = torch.ones_like(out_patch)

                    E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                    W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
            output = E.div_(W)

        return output
        
    def enhance(self, img_array):
        """
        Enhance an image using the DRCT model with tiling support
        
        Args:
            img_array: numpy array of the image (BGR format from cv2)
            
        Returns:
            Enhanced image as numpy array
        """
        logger.info("Processing image with DRCT model...")
        
        # Convert to RGB and normalize to [0, 1]
        img = img_array.astype(np.float32) / 255.
        
        # Convert from numpy to torch tensor and rearrange to BCHW
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Handle padding
            _, _, h_old, w_old = img.size()
            h_pad = (h_old // self.window_size + 1) * self.window_size - h_old
            w_pad = (w_old // self.window_size + 1) * self.window_size - w_old
            
            # Pad the image using flip (reflection padding)
            img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h_old + h_pad, :]
            img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w_old + w_pad]
            
            # Run inference with tiling
            output = self.test(img)
            
            # Crop to original size * scale
            output = output[..., :h_old * self.scale, :w_old * self.scale]
            
            # Convert back to numpy array
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # Convert back to BGR
            output = (output * 255.0).round().astype(np.uint8)
            
        return output

# Initialize the model
enhancement_model = ImageEnhancementModel()

@app.get("/")
async def root():
    return {"message": "Welcome to EnhanceIt API using DRCT model. Use /enhance endpoint to improve your images."}


@app.post("/enhance/")
async def enhance_image(file: UploadFile = File(...)):
    try:
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png", "image/webp", "image/jpg"]:
            raise HTTPException(status_code=400, detail="Invalid file format. Please upload a JPEG or PNG image.")
        
        # Read the image with OpenCV
        contents = await file.read()
        img_array = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Could not read image file")
        
        # Enhance the image using DRCT model
        enhanced_img = enhancement_model.enhance(img)
        
        # Convert the enhanced image to bytes
        _, buffer = cv2.imencode(Path(file.filename).suffix, enhanced_img)
        enhanced_bytes = io.BytesIO(buffer).read()
        
        # Return the enhanced image directly (no disk saving)
        return StreamingResponse(
            io.BytesIO(enhanced_bytes),
            media_type=f"image/{Path(file.filename).suffix[1:]}",
            headers={
                "Content-Disposition": f"attachment; filename=enhanced_{file.filename}"
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)