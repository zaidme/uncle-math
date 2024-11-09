from manim import Scene, config  # Import config explicitly
import math
import os
from api.manim_generator.working_script import EnhancedMathSolutionScene

def solve_problem(problem_text, test_mode=False):  # Change default to False
    try:
        print(f"Starting video generation in {'test' if test_mode else 'live'} mode...")
        
        # Configure manim
        config.quality = "medium_quality"
        config.preview = False
        
        print("Creating scene...")
        scene = EnhancedMathSolutionScene(
            problem=problem_text,
            avatar_path="assets/avatar_icon.png",
            test_mode=test_mode,  # Pass through the test_mode parameter
            test_data=None
        )
        
        print("Rendering scene...")
        scene.render()
        print("Scene rendering complete!")
        
        output_file = scene.renderer.file_writer.movie_file_path
        if output_file and os.path.exists(output_file):
            print(f"Video file created successfully at: {output_file}")
            return True
        else:
            raise Exception("Video file not found after rendering")
            
    except Exception as e:
        print(f"Error in solve_problem: {str(e)}")
        import traceback
        traceback.print_exc()
        raise