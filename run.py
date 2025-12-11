#!/usr/bin/env python3
"""Simple script to run the semantic detection service"""
import os
import uvicorn

if __name__ == "__main__":
    # Get port from environment variable (Render sets this) or default to 8001
    port = int(os.getenv("PORT", 8001))
    # Disable reload in production
    # Render sets RENDER=true, or we can check for other production indicators
    is_production = (
        os.getenv("RENDER") is not None or 
        os.getenv("ENVIRONMENT") == "production" or
        os.getenv("PYTHON_ENV") == "production"
    )
    reload = not is_production
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=reload,
        log_level="info"
    )

