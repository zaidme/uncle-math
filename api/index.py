from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import time
import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Get to project root
MANIM_OUTPUT_DIR = os.path.join(BASE_DIR, "media", "videos", "720p30")
API_MEDIA_DIR = os.path.join(BASE_DIR, "api", "media")

# Create directories if they don't exist
os.makedirs(API_MEDIA_DIR, exist_ok=True)
app.mount("/api/media", StaticFiles(directory=API_MEDIA_DIR), name="media")

class ProblemRequest(BaseModel):
    problem_text: str
    isTestMode: bool = True

@app.post("/api/generate-video")
async def generate_video(request: ProblemRequest):
    try:
        # Source and destination paths
        manim_output = os.path.join(MANIM_OUTPUT_DIR, "EnhancedMathSolutionScene.mp4")
        output_filename = "math_solution_test.mp4" if request.isTestMode else f"math_solution_{hash(request.problem_text)}.mp4"
        final_path = os.path.join(API_MEDIA_DIR, output_filename)
        
        print(f"Starting video generation...")
        print(f"Expected manim output: {manim_output}")
        print(f"Final destination: {final_path}")
        
        # Generate the video
        from api.manim_generator.run_scene import solve_problem
        solve_problem(
            problem_text=request.problem_text if not request.isTestMode else "test problem",
            test_mode=request.isTestMode
        )
        
        # Wait for the file to appear and copy it
        max_wait = 30
        wait_interval = 0.5
        elapsed = 0
        
        while elapsed < max_wait:
            if os.path.exists(manim_output) and os.path.getsize(manim_output) > 0:
                try:
                    shutil.copy2(manim_output, final_path)
                    print(f"Successfully copied video to {final_path}")
                    break
                except Exception as e:
                    print(f"Error copying file: {e}")
            time.sleep(wait_interval)
            elapsed += wait_interval
            print(f"Waiting for video file... {elapsed}s")
            
        if not os.path.exists(final_path):
            raise HTTPException(
                status_code=500,
                detail=f"Video generation failed - file not found at {manim_output}"
            )
            
        # Verify file size
        if os.path.getsize(final_path) == 0:
            raise HTTPException(
                status_code=500,
                detail="Video generation failed - file is empty"
            )
            
        return {
            "status": "success",
            "videoUrl": f"/api/media/{output_filename}",
            "message": "Video generated successfully"
        }
        
    except Exception as e:
        print(f"Error generating video: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Add directory contents to error message for debugging
        manim_files = os.listdir(MANIM_OUTPUT_DIR) if os.path.exists(MANIM_OUTPUT_DIR) else []
        api_media_files = os.listdir(API_MEDIA_DIR) if os.path.exists(API_MEDIA_DIR) else []
        
        error_detail = {
            "error": str(e),
            "manim_output_dir": MANIM_OUTPUT_DIR,
            "manim_files": manim_files,
            "api_media_dir": API_MEDIA_DIR,
            "api_media_files": api_media_files
        }
        
        raise HTTPException(
            status_code=500,
            detail=str(error_detail)
        )

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/api/debug/paths")
async def debug_paths():
    """Endpoint to debug file paths and directory contents"""
    return {
        "base_dir": BASE_DIR,
        "manim_output_dir": MANIM_OUTPUT_DIR,
        "api_media_dir": API_MEDIA_DIR,
        "manim_files": os.listdir(MANIM_OUTPUT_DIR) if os.path.exists(MANIM_OUTPUT_DIR) else [],
        "api_media_files": os.listdir(API_MEDIA_DIR) if os.path.exists(API_MEDIA_DIR) else [],
        "current_working_dir": os.getcwd()
    }