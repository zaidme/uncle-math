from manim import *
from anthropic import Anthropic
from dotenv import load_dotenv
from PIL import Image
import os
import json
import math
import numpy as np

# Load environment variables
load_dotenv()

class MathProblemSolver:
    def __init__(self, test_mode=False, test_data=None):
        self.test_mode = test_mode
        self.test_data = test_data
        if not test_mode:
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
            self.client = Anthropic(api_key=api_key)

    def get_solution(self, problem):
        """Get solution steps from Claude API or test data"""
        if self.test_mode:
            print("Running in test mode")
            test_data = test_data = {
    "problem_type": "calculus",
    "initial_problem": "why is e^x differentiated still e^x",
    "steps": [
        {
            "equation": "y = e^x",
            "explanation": "Let's start with the basic exponential function",
            "actions": [
                {
                    "type": "graph",
                    "params": {
                        "function": "2.718281828459045**x",
                        "x_range": [-2, 2, 0.5],
                        "y_range": [0, 8, 1]
                    }
                }
            ]
        },
        {
            "equation": "\\frac{d}{dx}[e^x] = \\lim_{h \\to 0} \\frac{e^{x+h} - e^x}{h}",
            "explanation": "Using the definition of derivative",
            "actions": [
                {
                    "type": "highlight",
                    "params": {
                        "target": "\\lim_{h \\to 0}",
                        "color": "BLUE"
                    }
                }
            ]
        },
        {
            "equation": "= \\lim_{h \\to 0} \\frac{e^x(e^h - 1)}{h}",
            "explanation": "Factor out e^x from numerator",
            "actions": [
                {
                    "type": "highlight",
                    "params": {
                        "target": "e^x",
                        "color": "GREEN"
                    }
                }
            ]
        },
        {
            "equation": "= e^x \\lim_{h \\to 0} \\frac{e^h - 1}{h}",
            "explanation": "Pull e^x out of limit as it's constant with respect to h",
            "actions": [
                {
                    "type": "highlight",
                    "params": {
                        "target": "\\lim_{h \\to 0} \\frac{e^h - 1}{h}",
                        "color": "RED"
                    }
                }
            ]
        },
        {
            "equation": "= e^x \\cdot 1",
            "explanation": "The limit evaluates to 1 (this is a fundamental limit)",
            "actions": [
                {
                    "type": "highlight",
                    "params": {
                        "target": "1",
                        "color": "YELLOW"
                    }
                }
            ]
        }
    ],
    "final_answer": "\\frac{d}{dx}[e^x] = e^x",
    "solution_context": "e^x is unique because its derivative equals itself. This happens because the exponential function e^x has a special property where its rate of change at any point is proportional to its value at that point, with a proportionality constant of 1. This is why e^x is so important in modeling natural phenomena where the rate of change is proportional to the current value, such as population growth or radioactive decay."
}
            return test_data
            
        try:
            prompt = f"""You are a math solution JSON generator. Respond ONLY with a JSON object, no other text. Analyze and provide solution for: {problem}

JSON structure must be exactly:
{{
    "problem_type": "algebraic/geometric/calculus/etc",
    "initial_problem": "{problem}",
    "steps": [
        {{
            "equation": "step equation",
            "explanation": "step explanation",
            "actions": [
                {{
                    "type": "action_name",
                    "params": {{
                        // parameters specific to action type
                    }}
                }}
            ]
        }}
    ],
    "final_answer": "final answer",
    "solution_context": "explanation of solution"
}}

Available actions:
- Basic Actions:
  "highlight": Highlight part of an equation
    params: {{"target": "term to highlight", "color": "color_name"}}
  "clear": Clear temporary visualizations
    params: {{"target": "all/graph/shapes/etc"}}

- Graph Actions:
  "graph": Create/update a graph
    params: {{
      "function": "function to plot",
      "x_range": [-5, 5, 1],
      "y_range": [-5, 10, 1],
      "points": [{"x": number, "y": number, "label": "label"}],
      "h_lines": [{"y": number, "label": "label"}],
      "discontinuities": [number]  // x-values where function is discontinuous
    }}
  "coordinate_lines": Add tracking lines to a point
    params: {{
      "x": number,
      "y": number,
      "label": "label",
      "show_coordinates": true/false
    }}
  "track_point": Create a point that tracks a function
    params: {{
      "function": "function to track",
      "x_value": number,
      "animate_to": number  // x value to animate to
    }}

- Transform Actions:
  "transform": Apply transformation to an object
    params: {{
      "target": "object_id",
      "type": "rotate/stretch/scale",
      "value": number,
      "axis": "x/y"  // for stretch
    }}
  "function_transform": Transform one function to another
    params: {{
      "from_function": "initial function",
      "to_function": "final function",
      "transition_time": number
    }}
  "complex_transform": Apply complex function transform
    params: {{
      "function": "z**2" // example complex function
    }}

- Text and Layout Actions:
  "text_gradient": Create gradient-colored text
    params: {{
      "text": "text to display",
      "colors": ["color1", "color2"],
      "direction": "horizontal/vertical"
    }}
  "grid_layout": Create a grid of objects
    params: {{
      "items": ["item1", "item2"],
      "rows": number,
      "cols": number
    }}

Rules:
1. Functions must be valid Python math (e.g., '2*x + 3' not '2x + 3')
2. All numeric values must be actual numbers
3. Each step must include at least one action
4. Colors should be from: BLUE, RED, GREEN, YELLOW, WHITE
5. Response must be ONLY the JSON object, no other text"""

            # Replace placeholder with actual problem
            prompt = prompt.replace("[PROBLEM]", problem)

            print(f"Sending request to Claude API for problem: {problem}")
            message = self.client.messages.create(
                model="claude-3-sonnet-20240229679679679696976",
                temperature=0,
                max_tokens=4096,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            # Extract and process response
            response_text = message.content[0].text if isinstance(message.content, list) else message.content
            
            # Try to find JSON content within the response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if not json_match:
                raise ValueError("No JSON found in Claude's response")
                
            json_str = json_match.group()
            
            try:
                solution_data = json.loads(json_str)
                
                # Validate required fields
                required_fields = ['problem_type', 'initial_problem', 'concept_introduction', 
                                'prerequisite_knowledge', 'steps', 'final_answer', 
                                'solution_context', 'common_mistakes', 'further_applications']
                
                missing_fields = [field for field in required_fields if field not in solution_data]
                if missing_fields:
                    raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
                
                # Validate steps structure
                for i, step in enumerate(solution_data['steps']):
                    required_step_fields = ['equation', 'explanation', 'conceptual_explanation', 
                                        'technical_explanation', 'actions']
                    missing_step_fields = [field for field in required_step_fields if field not in step]
                    if missing_step_fields:
                        raise ValueError(f"Step {i + 1} is missing required fields: {', '.join(missing_step_fields)}")
                
                return solution_data
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {str(e)}")
                print(f"Response text: {response_text}")
                raise ValueError("Failed to parse Claude's response as JSON")
                
        except Exception as e:
            print(f"Error getting solution from Claude: {str(e)}")
            if self.test_mode:
                print("Falling back to test mode due to error")
                return self.get_solution(problem)  # Recursive call in test mode
            raise ValueError(f"Failed to get solution from Claude: {str(e)}")

        finally:
            print("Finished processing solution request")
        
class EnhancedMathSolutionScene(Scene):
    def __init__(self, problem, avatar_path=None, test_mode=False, test_data=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.solver = MathProblemSolver(test_mode=test_mode, test_data=test_data)
        self.solution_data = self.solver.get_solution(problem)
        self.persistent_objects = {
            'graph': None,
            'shapes': [],
            'tracking_points': {},
            'coordinate_lines': {}
        }
        self.avatar_path = avatar_path or "assets/avatar_icon.png"

        # Reading speed configuration
        self.words_per_minute = 250
        self.min_wait_time = 1.5
        self.max_wait_time = 5
        self.math_multiplier = 1.5
        
        # Transition timing configuration
        self.equation_write_time = 1    # Time to write/transform equation
        self.transition_pause = 0.5     # Brief pause between elements

    def calculate_read_time(self, text, has_math=False):
        """Calculate appropriate wait time based on text length."""
        word_count = len(text.split())
        read_time = (word_count / (self.words_per_minute / 60))
        read_time *= 1.3  # Buffer time
        
        if has_math:
            read_time *= self.math_multiplier
        
        return max(min(read_time, self.max_wait_time), self.min_wait_time)

    
    def create_avatar(self):
        """Create an avatar circle with the image"""
        try:
            # Load and process the image
            image = Image.open(self.avatar_path)
            
            # Convert to RGBA if not already
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            
            # Make it square by cropping to the smaller dimension
            size = min(image.size)
            left = (image.size[0] - size) // 2
            top = (image.size[1] - size) // 2
            image = image.crop((left, top, left + size, top + size))
            
            # Resize to a reasonable size for the animation
            image = image.resize((256, 256), Image.Resampling.LANCZOS)
            
            # Convert PIL image to numpy array and create ImageMobject
            img_array = np.array(image)
            avatar_image = ImageMobject(img_array).scale(0.4)  # Scale to match circle size
            avatar_image.to_corner(DL, buff=0.7)

            return avatar_image  # Just return the ImageMobject directly

        except Exception as e:
            print(f"Error loading avatar image: {str(e)}")
            # Return a circle as fallback
            return Circle(
                radius=0.5,
                fill_opacity=1,
                fill_color=BLUE_E,
                stroke_color=LIGHT_GREY,
                stroke_width=2
            ).to_corner(DL, buff=0.7)
        
    def create_explanation_box(self, text):
        """Create a styled explanation box with text and avatar"""
        # Create the box with rounded corners
        box = RoundedRectangle(
            corner_radius=0.2,
            height=1.2,  # Slightly taller to accommodate text better
            width=9,
            fill_opacity=0.2,
            fill_color=DARK_GREY,
            stroke_color=LIGHT_GREY
        ).to_edge(DOWN, buff=0.5)

        # Load avatar once and reuse
        if not hasattr(self, 'avatar_image'):
            self.avatar_image = self.create_avatar()
            self.avatar_image.scale(0.8)  # Scale down a bit more

        # Position avatar on left side of box
        avatar_copy = self.avatar_image.copy()
        avatar_copy.next_to(box.get_left(), RIGHT, buff=0.3)
        
        # Calculate available width for text (box width minus avatar and padding)
        available_width = box.width - avatar_copy.width - 1  # 1 unit for padding
        
        # Create the text inside the box with fixed font size
        text_obj = Text(
            text,
            font_size=16,  # Fixed font size
            color=WHITE
        )
        
        # Scale text to fit if necessary
        if text_obj.width > available_width:
            text_obj.scale_to_fit_width(available_width)
        
        # Position text in center of remaining space
        text_obj.next_to(avatar_copy, RIGHT, buff=0.3)
        # Center the text vertically relative to the box
        text_obj.move_to([
            text_obj.get_center()[0],  # Keep x position
            box.get_center()[1],       # Use box's y position
            0                          # z position
        ])

        return Group(box, avatar_copy, text_obj)

    def create_tracking_point(self, params):
        """Create a point that tracks along a function with coordinate lines"""
        if not self.persistent_objects["graph"]:
            return None
            
        axes = self.persistent_objects["graph"][0]
        
        def func(x):
            return eval(params["function"].replace("x", str(x)))
        
        x_val = params["x_value"]
        y_val = func(x_val)
        
        # Create the dot at the point
        dot = Dot(color=RED)
        dot.move_to(axes.coords_to_point(x_val, y_val))
        
        if params.get("show_coordinates", False):
            # Create horizontal line
            def create_h_line():
                p1 = axes.coords_to_point(axes.x_range[0], y_val)
                p2 = axes.coords_to_point(x_val, y_val)
                return DashedLine(p1, p2, color=YELLOW)
            
            # Create vertical line
            def create_v_line():
                p1 = axes.coords_to_point(x_val, axes.y_range[0])
                p2 = axes.coords_to_point(x_val, y_val)
                return DashedLine(p1, p2, color=YELLOW)
            
            h_line = always_redraw(create_h_line)
            v_line = always_redraw(create_v_line)
            
            # Create coordinate label
            coord_label = always_redraw(
                lambda: Text(f"({x_val:.1f}, {y_val:.1f})")
                    .scale(0.5)
                    .next_to(dot, UR, buff=0.1)
            )
            
            return VGroup(dot, h_line, v_line, coord_label)
        
        return dot


    def create_graph(self, params):
        """Enhanced graph creation with discontinuities support"""
        axes = Axes(
            x_range=params.get("x_range", [-5, 5, 1]),
            y_range=params.get("y_range", [-5, 10, 1]),
            axis_config={"color": BLUE},
            x_length=5,
            y_length=5
        )
        
        graph_group = VGroup(axes)
        
        if "function" in params:
            discontinuities = params.get("discontinuities", [])
            
            def func(x):
                return eval(params["function"].replace("x", str(x)))
            
            # Split the graph at discontinuities
            if discontinuities:
                x_range = params.get("x_range", [-5, 5])
                segments = []
                points = sorted([x_range[0]] + discontinuities + [x_range[1]])
                
                for i in range(len(points) - 1):
                    start, end = points[i], points[i + 1]
                    segment = axes.plot(
                        func,
                        x_range=[start + 0.001, end - 0.001],
                        color=YELLOW
                    )
                    segments.append(segment)
                
                graph_group.add(*segments)
            else:
                graph = axes.plot(func, color=YELLOW)
                graph_group.add(graph)
            
            # Add points if specified
            if "points" in params:
                for point in params["points"]:
                    dot = Dot(
                        axes.coords_to_point(point["x"], point["y"]),
                        color=RED
                    )
                    if "label" in point:
                        label = MathTex(point["label"]).next_to(dot, UR, buff=0.1)
                        graph_group.add(VGroup(dot, label))
                    else:
                        graph_group.add(dot)
            
            # Add horizontal lines if specified
            if "h_lines" in params:
                for line in params["h_lines"]:
                    y = line["y"]
                    h_line = axes.get_horizontal_line(
                        axes.c2p(axes.x_range[1], y),
                        color=BLUE_D
                    )
                    if "label" in line:
                        label = MathTex(line["label"]).next_to(h_line, RIGHT)
                        graph_group.add(VGroup(h_line, label))
                    else:
                        graph_group.add(h_line)
        
        return graph_group.scale(0.5).to_edge(RIGHT)


    def handle_action(self, action, equation_obj):
        """Enhanced action handler with new actions"""
        """Enhanced action handler with new actions"""
        action_type = action["type"]
        params = action.get("params", {})
        
        if action_type == "highlight":
            target = params.get("target")
            color = globals().get(params.get("color", "YELLOW"))
            
            # Find the target term in the equation
            # We'll create a new equation with the target term highlighted
            highlighted_eq = equation_obj.copy()
            
            # If the equation is a MathTex object, we can highlight specific parts
            if isinstance(equation_obj, MathTex):
                # Find all substrings that match the target
                for i, part in enumerate(highlighted_eq.tex_strings):
                    if target in part:
                        highlighted_eq[i].set_color(color)
                
                # Create the animation sequence
                return AnimationGroup(
                    FadeOut(equation_obj),
                    FadeIn(highlighted_eq)
                )
            else:
                # If not a MathTex object, just change the entire color
                highlighted_eq.set_color(color)
                return ReplacementTransform(equation_obj, highlighted_eq)
            
        elif action_type == "graph":
            graph = self.create_graph(params)
            if self.persistent_objects["graph"] is None:
                self.persistent_objects["graph"] = graph
                return Create(graph)
            else:
                old_graph = self.persistent_objects["graph"]
                self.persistent_objects["graph"] = graph
                return ReplacementTransform(old_graph, graph)
                
        elif action_type == "clear":
            animations = []
            target = params.get("target", "all")
            if target in ["all", "shapes"]:
                for shape in self.persistent_objects["shapes"]:
                    animations.append(FadeOut(shape))
                self.persistent_objects["shapes"] = []
            return AnimationGroup(*animations) if animations else None
            
        elif action_type == "track_point":
            point = self.create_tracking_point(params)
            if point:
                self.persistent_objects["tracking_points"][params["function"]] = point
                if params.get("animate_to"):
                    x_start = params["x_value"]
                    x_end = params["animate_to"]
                    
                    def update_point(mob, alpha):
                        x = x_start + (x_end - x_start) * alpha
                        y = eval(params["function"].replace("x", str(x)))
                        mob.move_to(self.persistent_objects["graph"][0].c2p(x, y))
                    
                    return UpdateFromAlphaFunc(point, update_point)
                return Create(point)
                
        elif action_type == "function_transform":
            if not self.persistent_objects["graph"]:
                return None
                
            old_graph = self.persistent_objects["graph"]
            
            def create_function_graph(function_str):
                return self.create_graph({"function": function_str})
            
            new_graph = create_function_graph(params["to_function"])
            self.persistent_objects["graph"] = new_graph
            return ReplacementTransform(
                old_graph,
                new_graph,
                run_time=params.get("transition_time", 2)
            )
            
        
        
        elif action_type == "coordinate_lines":
            if not self.persistent_objects["graph"]:
                return None
                
            axes = self.persistent_objects["graph"][0]
            x_val = params["x"]
            y_val = params["y"]
            
            # Create the dot at the specified coordinates
            dot = Dot(color=RED)
            dot.move_to(axes.coords_to_point(x_val, y_val))
            
            lines_group = VGroup(dot)
            
            if params.get("show_coordinates", False):
                h_line = DashedLine(
                    axes.coords_to_point(axes.x_range[0], y_val),
                    axes.coords_to_point(x_val, y_val),
                    color=YELLOW
                )
                
                v_line = DashedLine(
                    axes.coords_to_point(x_val, axes.y_range[0]),
                    axes.coords_to_point(x_val, y_val),
                    color=YELLOW
                )
                
                lines_group.add(h_line, v_line)
                    
            if "label" in params:
                # Convert common mathematical symbols to LaTeX
                label_text = params["label"]
                # Common replacements
                replacements = {
                    "π": "\\pi",
                    "θ": "\\theta",
                    "√": "\\sqrt",
                    "∞": "\\infty",
                    "α": "\\alpha",
                    "β": "\\beta",
                    "Δ": "\\Delta",
                    "μ": "\\mu",
                    "σ": "\\sigma"
                }
                
                for symbol, latex in replacements.items():
                    label_text = label_text.replace(symbol, latex)
                    
                label = MathTex(label_text).next_to(dot, UR, buff=0.1)
                lines_group.add(label)
            
            self.persistent_objects["coordinate_lines"][f"{x_val},{y_val}"] = lines_group
            return Create(lines_group)
        
            
        elif action_type == "text_gradient":
            text = Text(
                params["text"],
                gradient=params.get("colors", [BLUE, GREEN]),
                direction=params.get("direction", "horizontal")
            )
            return Write(text)
            
        elif action_type == "grid_layout":
            # Handle grid layout
            return None  # Placeholder for grid layout implementation
            
        return None


    def construct(self):
        # Initial problem statement
        title = Text(
            self.solution_data['initial_problem'],
            color=WHITE,
            font_size=16
        ).to_edge(UL)
        
        self.play(Write(title))
        self.wait(self.calculate_read_time(self.solution_data['initial_problem'], has_math=True))

        current_eq = None
        current_exp_box = None
        equation_group = VGroup().shift(UP * 2).to_edge(LEFT, buff=1)  # Container for equations
        
        for step in self.solution_data["steps"]:
            # Clear previous step's content if it exists
            if current_eq:
                self.remove(current_eq)
            if current_exp_box:
                self.remove(current_exp_box)
                
            # Create new equation and explanation
            new_eq = MathTex(step["equation"], color=WHITE)
            new_eq.move_to(equation_group)  # Position equation in designated area
            
            new_exp_box = self.create_explanation_box(step["explanation"])
            
            # Phase 1: Handle equation transition
            if current_eq is None:
                self.play(Write(new_eq), run_time=self.equation_write_time)
            else:
                self.play(
                    ReplacementTransform(current_eq, new_eq),
                    run_time=self.equation_write_time
                )
            
            # Wait for equation to be read
            eq_read_time = self.calculate_read_time(step["equation"], has_math=True)
            self.wait(eq_read_time)
            
            # Phase 2: Bring in explanation
            if current_exp_box is None:
                self.play(
                    FadeIn(new_exp_box[0]),  # Box
                    FadeIn(new_exp_box[1]),  # Avatar
                    Write(new_exp_box[2]),   # Text
                    run_time=1.5
                )
            else:
                self.play(
                    ReplacementTransform(current_exp_box[0], new_exp_box[0]),  # Box
                    ReplacementTransform(current_exp_box[1], new_exp_box[1]),  # Avatar
                    ReplacementTransform(current_exp_box[2], new_exp_box[2]),  # Text
                    run_time=1.5
                )
            
            # Wait for explanation to be read
            exp_read_time = self.calculate_read_time(step["explanation"])
            self.wait(exp_read_time)
            
            # Phase 3: Handle animations for this step
            action_animations = []
            for action in step.get("actions", []):
                # Clear previous graph if we're about to create a new one
                if action["type"] == "graph" and self.persistent_objects["graph"]:
                    self.remove(self.persistent_objects["graph"])
                    self.persistent_objects["graph"] = None
                    
                action_animation = self.handle_action(action, new_eq)
                if action_animation:
                    action_animations.append(action_animation)
            
            if action_animations:
                self.play(*action_animations, run_time=2)
                self.wait(0.5)  # Brief pause after actions
            
            current_eq = new_eq
            current_exp_box = new_exp_box
            
            # Brief pause before next step
            self.wait(self.transition_pause)

        # Create final explanation combining final answer and context
        final_text = f"Final Answer: {self.solution_data['final_answer']}\n\n{self.solution_data['solution_context']}"
        final_exp_box = self.create_explanation_box(final_text)
        
        # Clean up previous objects
        if current_eq:
            self.play(FadeOut(current_eq))
        if current_exp_box:
            self.play(
                ReplacementTransform(current_exp_box[0], final_exp_box[0]),  # Box
                ReplacementTransform(current_exp_box[1], final_exp_box[1]),  # Avatar
                ReplacementTransform(current_exp_box[2], final_exp_box[2])   # Text
            )
        
        # Wait for final explanation and answer to be read
        final_read_time = self.calculate_read_time(final_text, has_math=True)
        self.wait(final_read_time)
        
        # Clean up
        self.play(
            FadeOut(title),
            FadeOut(final_exp_box),
            FadeOut(self.persistent_objects["graph"]) if self.persistent_objects["graph"] else None
        )