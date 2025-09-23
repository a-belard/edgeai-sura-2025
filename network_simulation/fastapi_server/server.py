#!/usr/bin/env python3
"""
FastAPI Server for SqueezeNet Model Inference
Hosts the model for real RTT simulation testing.
"""

import os
import sys
import time
import io
import base64
from typing import Optional, Dict, Any
import json
import asyncio
import logging
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request/Response Models
class InferenceRequest(BaseModel):
    image_data: str  # base64 encoded image
    batch_size: Optional[int] = 1
    include_timing: Optional[bool] = True

class InferenceResponse(BaseModel):
    predictions: list
    confidence_scores: list
    processing_time_ms: float
    model_info: Dict[str, Any]
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_info: Dict[str, Any]

# Global model instance
model_instance = None
device = None
model_info = {}
transform = None

class SqueezeNetServer:
    def __init__(self, use_quantized: bool = False):
        self.model = None
        self.device = None
        self.use_quantized = use_quantized
        self.model_size_mb = 0
        self.load_time = 0
        
    def setup(self):
        """Initialize the model and device"""
        start_time = time.time()
        
        # Device selection
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
            
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = models.squeezenet1_1(pretrained=True)
        self.model.eval()
        self.model = self.model.to(self.device)
        
        # Apply quantization if requested
        if self.use_quantized:
            self.model = torch.quantization.quantize_dynamic(
                self.model, 
                {torch.nn.Linear, torch.nn.Conv2d}, 
                dtype=torch.qint8
            )
            
        # Calculate model size
        self.model_size_mb = sum(
            p.nelement() * p.element_size() for p in self.model.parameters()
        ) / (1024 * 1024)
        
        self.load_time = time.time() - start_time
        
        logger.info(f"Model loaded in {self.load_time:.2f}s, size: {self.model_size_mb:.2f}MB")
        
    def preprocess_image(self, image_data: str) -> torch.Tensor:
        """Convert base64 image to model input tensor"""
        try:
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Transform
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            tensor = transform(image).unsqueeze(0)  # Add batch dimension
            return tensor.to(self.device)
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {str(e)}")
    
    def inference(self, input_tensor: torch.Tensor) -> tuple:
        """Run inference and return predictions with timing"""
        start_time = time.perf_counter()
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            
        # Synchronize if using CUDA
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            
        end_time = time.perf_counter()
        processing_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Get top 5 predictions
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top5_prob, top5_indices = torch.topk(probabilities, 5)
        
        predictions = top5_indices.cpu().numpy().tolist()
        confidence_scores = top5_prob.cpu().numpy().tolist()
        
        return predictions, confidence_scores, processing_time

# Initialize FastAPI app
app = FastAPI(
    title="SqueezeNet Inference Server",
    description="FastAPI server hosting SqueezeNet for real RTT simulation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize server on startup
@app.on_event("startup")
async def startup_event():
    global model_instance
    try:
        model_instance = SqueezeNetServer(use_quantized=False)
        model_instance.setup()
        
        global model_info
        model_info = {
            "name": "SqueezeNet 1.1",
            "size_mb": model_instance.model_size_mb,
            "device": str(model_instance.device),
            "quantized": model_instance.use_quantized,
            "load_time_s": model_instance.load_time
        }
        
        logger.info("Server startup complete")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="online",
        model_loaded=model_instance is not None,
        device=str(model_instance.device) if model_instance else "unknown",
        model_info=model_info
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    return HealthResponse(
        status="healthy" if model_instance else "error",
        model_loaded=model_instance is not None,
        device=str(model_instance.device) if model_instance else "unknown",
        model_info=model_info
    )

@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    """Main inference endpoint"""
    if not model_instance:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Preprocess image
        input_tensor = model_instance.preprocess_image(request.image_data)
        
        # Run inference
        predictions, confidence_scores, processing_time = model_instance.inference(input_tensor)
        
        return InferenceResponse(
            predictions=predictions,
            confidence_scores=confidence_scores,
            processing_time_ms=processing_time,
            model_info=model_info,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    """Inference endpoint for file uploads"""
    if not model_instance:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read and encode file
        content = await file.read()
        image_data = base64.b64encode(content).decode('utf-8')
        
        # Create request
        request = InferenceRequest(image_data=image_data)
        
        # Process
        return await predict(request)
        
    except Exception as e:
        logger.error(f"File inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File inference failed: {str(e)}")

@app.get("/model_info")
async def get_model_info():
    """Get detailed model information"""
    if not model_instance:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_info": model_info,
        "device_info": {
            "device": str(model_instance.device),
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
        },
        "server_info": {
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "startup_time": time.time()
        }
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SqueezeNet FastAPI Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--quantized", action="store_true", help="Use quantized model")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    # Update quantization setting
    if args.quantized:
        # This would need to be handled in startup_event
        pass
        
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )