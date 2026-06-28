import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.core import config
from app.api import prediction_routes, insights_routes

# Load environment variables
load_dotenv()

# Initialize FastAPI application
app = FastAPI(
    title="GlowWise AI API",
    description="FastAPI backend for skincare review satisfaction prediction and ML insights.",
    version="0.1.0"
)

# Configure CORS origins using config values
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount APIRouters
app.include_router(prediction_routes.router)
app.include_router(insights_routes.router)

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Service health check endpoint.
    Returns status, environment debug flag, and service name details.
    """
    return {
        "status": "healthy",
        "service": "GlowWise AI Backend",
        "version": "0.1.0",
        "debug": os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")
    }

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    debug_mode = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")
    
    print(f"Starting GlowWise AI Backend on {host}:{port} with debug={debug_mode}...")
    uvicorn.run("main:app", host=host, port=port, reload=debug_mode)
