import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI application
app = FastAPI(
    title="GlowWise AI API",
    description="Skincare review intelligence ML inference and analytics API",
    version="1.0.0"
)

# Configure CORS origins
allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000")
allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Service health check endpoint.
    Returns status, environment debug flag, and service name details.
    """
    return {
        "status": "healthy",
        "service": "GlowWise AI Backend",
        "version": "1.0.0",
        "debug": os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")
    }

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    debug_mode = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")
    
    print(f"Starting GlowWise AI Backend on {host}:{port} with debug={debug_mode}...")
    uvicorn.run("main:app", host=host, port=port, reload=debug_mode)
