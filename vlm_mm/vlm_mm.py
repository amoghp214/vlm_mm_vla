import time
import io
import re

import requests
import torch
import json
from PIL import Image
from pathlib import Path
from transformers import AutoProcessor

from vlm_loaders import load_llava, load_gemma3, load_t5gemma
from vlm_utils import configure_fast_inference
import cv2
import imageio

class VLMMM:
    def __init__(
            self,
            model_name="gemma3",
            vlm_mm_context_dir="./path/to/vlm_mm_context/directory",
            vlm_mm_prompts_dir="./path/to/vlm_mm_prompts/directory",
            vla_original_prompt="the prompt that the VLA was given to start with",
            vla_curr_image_path="./path/to/vla_starting_image.jpg",
            vla_context_video_path="./path/to/vla_current_context.mp4",
            vlm_device: str | None = None,
        ):
        self.model_name = model_name
        self.vlm_mm_context_dir = Path(vlm_mm_context_dir)
        self.vlm_mm_prompts_dir = Path(vlm_mm_prompts_dir)
        self.vla_original_prompt = vla_original_prompt
        self.vla_curr_image_path = Path(vla_curr_image_path)
        self.vla_context_video_path = Path(vla_context_video_path)
        # requested device for the VLM model (string like "cuda:0" or "cpu")
        self.vlm_device = vlm_device
        
        configure_fast_inference()
        if model_name == "gemma3":
            self.model, self.processor = load_gemma3(device=self.vlm_device)
        elif model_name == "llava":
            self.model, self.processor = load_llava(device=self.vlm_device)
        elif model_name == "t5gemma":
            self.model, self.processor = load_t5gemma(device=self.vlm_device)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")


        # Used to index the mm_history: [(subtask #, task_history_step #), ...]
        # This will be updated whenever smaller tasks are created and when we navigate through the mm_history to store mm_curr_step_outputs
        # The 0th index corresponds to the original task
        self.curr_task_prefix = [(0, 0)]

        # Create the mm_history.json and task_hierarchy.json files
        mm_history_path = self.vlm_mm_context_dir / "mm_history.json"
        task_hierarchy_path = self.vlm_mm_context_dir / "task_hierarchy.json"

        assert not mm_history_path.exists(), "mm_history.json already exists. Please delete or move this file before initializing the VLMMM class to avoid overwriting previous history."
        assert not task_hierarchy_path.exists(), "task_hierarchy.json already exists. Please delete or move this file before initializing the VLMMM class to avoid overwriting previous history."

        if not mm_history_path.exists():
            with open(mm_history_path, "w") as f:
                json.dump({
                    "task_objective": self.vla_original_prompt,
                    "task_history": []
                }, f, indent=2, ensure_ascii=False)

        if not task_hierarchy_path.exists():
            with open(task_hierarchy_path, "w") as f:
                json.dump({
                    "task_objective": self.vla_original_prompt,
                    "finished": False
                }, f, indent=2, ensure_ascii=False)    

    
    def create_task_plan(self, debug=False):
        """
        Create a task plan based on the VLA's original prompt
        and starting image. This plan will be saved in task_hierarchy.json 
        and will be used to guide the VLA's actions.
        """
        # Load task planning prompt schema from file
        task_planning_system_prompt_path = Path(self.vlm_mm_prompts_dir) / "task_planning_system_prompt.txt"
        with open(task_planning_system_prompt_path, "r") as f:
            task_planning_system_prompt = f.read().strip()

        # Load inital robot image
        initial_robot_image = Image.open(self.vla_curr_image_path).convert("RGB")
        
        task_planning_prompt = [
            {
                "role": "system", 
                "content": [
                    {"type": "text", "text": task_planning_system_prompt}
                ]
            },
            {
                "role": "user", "content": [
                    {"type": "image", "image": initial_robot_image},
                    {"type": "text", "text": self.vla_original_prompt},
                ]
            },
        ]

        start_time = time.time()

        # Process inputs for VLM to generate task plan
        inputs = self.processor.apply_chat_template(
            task_planning_prompt,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            # padding="max_length",
            # truncation=True,
            # max_length=512,
        ).to(self.model.device)

        inputs_processed_time = time.time()

        # Generate task plan using VLM
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                # cache_implementation="static",
            )

        generation_completed_time = time.time()

        if (debug):
            print(f"Inputs processed in {inputs_processed_time - start_time:.2f} seconds")
            print(f"Generation completed in {generation_completed_time - inputs_processed_time:.2f} seconds")

        if getattr(self.model.config, "is_encoder_decoder", False):
            generated_ids = output[0]
        else:
            prompt_length = inputs.input_ids.shape[1]
            generated_ids = output[0][prompt_length:]

        task_plan = self.processor.decode(generated_ids, skip_special_tokens=True).strip()

        # Parse task plan into list of tasks (split by periods)
        task_plan_list = [task.strip() + "." for task in task_plan.split('.') if task.strip()]
        task_hierarchy = dict()
        task_hierarchy["task_objective"] = self.vla_original_prompt
        task_hierarchy["finished"] = False
        task_hierarchy["subtasks"] = []
        for task in task_plan_list:
            task_hierarchy["subtasks"].append({
                "task_objective": task,
                "finished": False
            })
        self.task_plan = task_plan_list
        self.curr_task = 0

        # Save task plan to file
        task_plan_json_path = self.vlm_mm_context_dir / "task_hierarchy.json"
        with open(task_plan_json_path, "w", encoding="utf-8") as f:
            json.dump(task_hierarchy, f, indent=2, ensure_ascii=False)
        
        if (debug):
            print(f"Generated task plan:\n{task_plan}")
        
        return task_plan_list

    def step(self, debug=False):
        """
        The VLM will determine the course of action for the VLA's
        prompt, the override action, whether to override the VLA's
        actions or not, and the reasoning behind these decisions. 
        This data will be saved as part of the VLM MM history.
        """
        # Step 1: Get inputs for mm current step
        # mm_curr_step_inputs = self.get_mm_curr_step_inputs()
        # curr_vla_step = mm_curr_step_inputs["vla_history"][-1]["step_num"] + 1

        # Step 2: Load:
        #   - mm current step reasoning prompt schema
        #   - image of the VLA and environment at the current step
        #   - current subtask objective for the VLA
        #   - VLA action history [-t:]
        #   - MM action history [f(-t):]
        #   - VLA environment trajectory video [-t:]
        mm_current_step_reasoning_system_prompt_path = self.vlm_mm_prompts_dir / "mm_step_prompt.txt"
        with open(mm_current_step_reasoning_system_prompt_path, "r") as f:
            mm_current_step_reasoning_system_prompt = f.read().strip()

        mm_inputs = self.get_mm_curr_step_inputs(debug=debug)
        curr_step_img = Image.open(mm_inputs["current_subtask_input_image"]).convert("RGB")
        
        # Step 3: Format the prompt with the inputs from Step 1
        mm_current_step_reasoning_prompt = [
            {
                "role": "system", 
                "content": [
                    {"type": "text", "text": mm_current_step_reasoning_system_prompt}
                ]
            },
            {
                "role": "user", "content": [
                    {"type": "image", "image": curr_step_img},
                    {"type": "text", "text": "The VLA's current subtask objective is: " + mm_inputs["current_subtask_objective"]},
                    {"type": "text", "text": f"The MM's history json: \n```json\n{json.dumps(mm_inputs['mm_history'], indent=2)}\n```"}
                ]
            },
        ]

        # Step 4: Pass the prompt through the VLM MM to get the outputs for the current step
        start_time = time.time()

        inputs = self.processor.apply_chat_template(
            mm_current_step_reasoning_prompt,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            # padding="max_length",
            # truncation=True,
            # max_length=512,
        ).to(self.model.device)

        inputs_processed_time = time.time()

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                # cache_implementation="static",  # Was causing issues since we use the model for 2 different calls
            )

        generation_completed_time = time.time()

        if (debug):
            print(f"MM Step inputs processed in {inputs_processed_time - start_time:.2f} seconds")
            print(f"MM Step generation completed in {generation_completed_time - inputs_processed_time:.2f} seconds")

        if getattr(self.model.config, "is_encoder_decoder", False):
            generated_ids = output[0]
        else:
            prompt_length = inputs.input_ids.shape[1]
            generated_ids = output[0][prompt_length:]

        next_action = self.processor.decode(generated_ids, skip_special_tokens=True).strip()
        if (debug): print(f"Generated next action:\n{next_action}")

        # Parse JSON from the model output and convert to dict for saving.
        try:
            curr_mm_outputs = self._extract_json_from_text(next_action)
            if ("num_steps_to_spend" not in curr_mm_outputs.keys()):
                curr_mm_outputs["num_steps_to_spend"] = 30  # default value if not provided by the model
            if (debug): print("Parsed curr_mm_outputs from VLM output.")
        except Exception as e:
            if (debug): print(f"Failed to parse JSON from VLM output: {e}. Using fallback no-evidence JSON.")
            curr_mm_outputs = {
                "is_current_subtask_done": False,
                "is_robot_struggling": False,
                "use_revised_prompt": False,
                "revised_prompt": "",
                "use_override_action": False,
                "override_action": [],
                "create_smaller_tasks": False,
                "smaller_tasks_list": [],
                "num_steps_to_spend": 15
            }
        if (debug): print("curr_mm_outputs:", curr_mm_outputs)

        # Step 6: Save the VLM MM's outputs for the current step
        self.save_mm_curr_step_outputs(
            # curr_step=curr_vla_step,
            curr_mm_outputs=curr_mm_outputs
        )
        
        # Return the outputs so external callers (e.g. the HTTP server) get the parsed results
        return curr_mm_outputs
    
    def get_mm_curr_step_inputs(self, debug=False):
        """
        Get the inputs for the VLM to determine the current step. This will include:
            - Current subtask objective
            - Current subtask input image
            - MM History of the past tasks, subtasks, revised prompts, actions, etc.

        Returns:
            dict: current_subtask_objective, current_subtask_input_image, mm_history
        """
        # Step 1: Load MM history and task hierarchy from file
        mm_history_path = self.vlm_mm_context_dir / "mm_history.json"
        task_hierarchy_path = self.vlm_mm_context_dir / "task_hierarchy.json"
        
        with open(mm_history_path, "r", encoding="utf-8") as f:
            mm_history = json.load(f)
        with open(task_hierarchy_path, "r", encoding="utf-8") as f:
            task_hierarchy = json.load(f)
        
        # Step 2: Get current subtask objective
        if (debug): print("Current task prefix:", self.curr_task_prefix)
        current_subtask_objective = self.get_subtask(self.curr_task_prefix, task_hierarchy)["task_objective"]

        # Step 3: Get current subtask input image
        current_subtask_input_image = self.vla_curr_image_path  # TODO: get this image in file so that it does not get overwritten and cause race conditions?

        return {
            "mm_history": mm_history,
            "current_subtask_objective": current_subtask_objective,
            "current_subtask_input_image": current_subtask_input_image,
        }

    # def save_mm_curr_step_outputs(self, curr_step, curr_mm_outputs):
    def save_mm_curr_step_outputs(self, curr_mm_outputs):
        """
        Save the VLM MM's outputs. This will include:
            - Status of current subtask (e.g. "completed", "not_started")
            - New set of inputs for the VLA:
                - Revised prompt (if generated)
                - Override action (if generated)
                - Smaller subtasks (if generated)
        """
        mm_history_path = self.vlm_mm_context_dir / "mm_history.json"
        task_hierarchy_path = self.vlm_mm_context_dir / "task_hierarchy.json"
        # mm_outputs_path = self.vlm_mm_context_dir / "mm_outputs.json"

        assert list(curr_mm_outputs.keys()) == [
            "is_current_subtask_done", 
            "is_robot_struggling", 
            "use_revised_prompt", 
            "revised_prompt",  
            "use_override_action", 
            "override_action", 
            "create_smaller_tasks", 
            "smaller_tasks_list",
            "num_steps_to_spend"
        ], f"curr_mm_outputs must contain keys: is_current_subtask_done, is_robot_struggling, use_revised_prompt, revised_prompt, use_override_action, override_action, create_smaller_tasks, smaller_tasks_list. \n\nKeys found: {list(curr_mm_outputs.keys())}"

        with open(mm_history_path, "r", encoding="utf-8") as f:
            mm_history = json.load(f)

        with open(task_hierarchy_path, "r", encoding="utf-8") as f:
            task_hierarchy = json.load(f)
        
        mm_output_parsed = dict()
        if (curr_mm_outputs["is_current_subtask_done"]):
            mm_output_parsed["mm_output_type"] = "progress"
        elif (not curr_mm_outputs["is_robot_struggling"]):
            # use previous prompt and action - TODO: should I have an explicit output type for this?
            mm_output_parsed["mm_output_type"] = "continue"
        elif (curr_mm_outputs["is_robot_struggling"] and curr_mm_outputs["use_revised_prompt"]):
            mm_output_parsed["mm_output_type"] = "revised_prompt"
            mm_output_parsed["output"] = curr_mm_outputs["revised_prompt"]
        elif (curr_mm_outputs["is_robot_struggling"] and curr_mm_outputs["use_override_action"]):
            mm_output_parsed["mm_output_type"] = "override_action"
            mm_output_parsed["output"] = curr_mm_outputs["override_action"]
        elif (curr_mm_outputs["is_robot_struggling"] and curr_mm_outputs["create_smaller_tasks"]):
            mm_output_parsed["mm_output_type"] = "smaller_tasks"
            mm_output_parsed["subtasks"] = curr_mm_outputs["smaller_tasks_list"][0]
            self.update_task_hierarchy(self.curr_task_prefix, curr_mm_outputs["smaller_tasks_list"])
        
        mm_output_parsed["num_steps"] = curr_mm_outputs.get("num_steps_to_spend", 30)
        
        # Insert mm_output_parsed into the correct part of mm_history.
        curr_subtask = self.get_curr_subtask_mm_history(mm_history)
        curr_subtask["task_history"].append(mm_output_parsed)

        # Update curr_task_prefix based on the mm_output_type
        if (mm_output_parsed["mm_output_type"] == "progress"):
            curr_task = self.get_subtask(self.curr_task_prefix, task_hierarchy)
            curr_task["finished"] = True
        self.curr_task_prefix = self.get_next_step_task_prefix(self.curr_task_prefix, task_hierarchy, mm_output_parsed["mm_output_type"])

        with open(mm_history_path, "w", encoding="utf-8") as f:
            json.dump(mm_history, f, indent=2, ensure_ascii=False)

    def get_curr_subtask_mm_history(self, mm_history):

        # Load mm_history from file
        # mm_history_path = self.vlm_mm_context_dir / "mm_history.json"
        # with open(mm_history_path, "r", encoding="utf-8") as f:
        #     mm_history = json.load(f)
        
        # Get current subtask from mm_history using self.curr_task_prefix
        if (len(self.curr_task_prefix) == 1): return mm_history

        tasks_list = mm_history["task_history"][self.curr_task_prefix[0][1]]["subtasks"]
        curr_task, curr_task_objective = None, None
        for t, h in self.curr_task_prefix[1:]:
            curr_task = tasks_list[t]
            curr_task_objective = curr_task["task_objective"]
            if ("subtasks" not in curr_task["task_history"][h].keys()):
                break
            tasks_list = curr_task["task_history"][h]["subtasks"]
        
        return curr_task
    
    def get_subtask(self, subtask_prefix, task_hierarchy):
        assert subtask_prefix[0][0] == 0, "The first element of subtask_prefix should be (0, x) corresponding to the original task"
        
        subtask_prefix = subtask_prefix[1:]  # remove the (0,x) prefix that corresponds to the original task
        if (subtask_prefix == []):  # The original task is the current subtask
            return task_hierarchy
        
        super_task = task_hierarchy["subtasks"]
        curr_task, curr_task_objective = None, None
        for t, _ in subtask_prefix:
            curr_task = super_task[t]
            curr_task_objective = curr_task["task_objective"]
            if ("subtasks" not in curr_task.keys()):
                break
            super_task = curr_task["subtasks"]
        
        return curr_task
    
    def get_next_step_task_prefix(self, curr_task_prefix, task_hierarchy, mm_output_type=None, debug=False):
        """
        Get the next step task prefix based on the current task prefix and the task hierarchy. This will involve:
            - Traversing the task hierarchy based on the current task prefix to find the current task and subtask
            - Determining the next task prefix based on the current task prefix and the structure of the task hierarchy
        """
        def r_get_next_subtask(c_prefix, c_task):
            if (debug): print("c_prefix:", c_prefix, c_task["finished"])
            if (c_prefix == []):
                return None
            subtask_next_prefix = r_get_next_subtask(c_prefix[1:], c_task["subtasks"][c_prefix[0][0]])
            if (subtask_next_prefix is not None):
                return [c_prefix[0]] + subtask_next_prefix
            if (c_task["subtasks"][c_prefix[0][0]]["finished"] == False):
                return [(c_prefix[0][0], c_prefix[0][1] + 1)]
            if not (c_prefix[0][0] + 1 < len(c_task["subtasks"])):
                return None
            return [(c_prefix[0][0] + 1, 0)]
        
        curr_task = self.get_subtask(curr_task_prefix, task_hierarchy)
        if (debug): print("curr_task finished:", curr_task["finished"])
        # If the task has not been finished, just increment the subtask history index
        if (curr_task["finished"] == False and mm_output_type != "smaller_tasks"):
            if (debug): print("curr_task not finished, incrementing subtask index")
            next_task_prefix = curr_task_prefix[:-1] + [(curr_task_prefix[-1][0], curr_task_prefix[-1][1] + 1)]
            return next_task_prefix
        
        # If the task has not been finished but new subtasks have been created, go to the first unfinished subtask
        if (curr_task["finished"] == False and mm_output_type == "smaller_tasks"):
            if (debug): print("curr_task not finished and new subtasks created, going to first unfinished subtask")
            for t, subtask in enumerate(curr_task["subtasks"]):
                if (subtask["finished"] == False):
                    next_task_prefix = curr_task_prefix + [(t, 0)]
                    return next_task_prefix
            assert False, "If new subtasks have been created, there should be at least one unfinished subtask."

        # If the task has been finished, find the next subtask that has not been finished
        next_task_partial_prefix = r_get_next_subtask(curr_task_prefix[1:], task_hierarchy)
        if (debug): print("next_task_partial_prefix:", next_task_partial_prefix)
        next_super_task_prefix = [curr_task_prefix[0]] + (next_task_partial_prefix if next_task_partial_prefix is not None else [])
        if (debug): print("next_super_task_prefix:", next_super_task_prefix)
        if (next_super_task_prefix == [curr_task_prefix[0]] and task_hierarchy["finished"] == True):
            return None

        # c_task = task_hierarchy
        # for t, _ in next_super_task_prefix:
        #     c_task = c_task["subtasks"][t]

        next_task_prefix = next_super_task_prefix

        # NOTE: In practice, this should not happen because the VLM would not have created these next tasks yet.
        # while ("subtasks" in c_task and len(c_task["subtasks"]) > 0):
        #     next_task_prefix.append((0, 0))
        #     c_task = c_task["subtasks"][0]
        return next_task_prefix

                

    def update_task_hierarchy(self, super_task_prefix, smaller_tasks_list, debug=False):
        """
        Update the task hierarchy JSON with the new smaller tasks. This will involve:
            - Finding the correct location in the task hierarchy to insert the smaller tasks based on the current task and subtask
            - Inserting the smaller tasks into the hierarchy
            - Saving the updated hierarchy back to the JSON file
        """
        if (debug): print(f"Updating task hierarchy at prefix {super_task_prefix} with smaller tasks: {smaller_tasks_list}")
        if (debug): print("Loading current task hierarchy from file...")
        task_hierarchy_path = self.vlm_mm_context_dir / "task_hierarchy.json"
        with open(task_hierarchy_path, "r", encoding="utf-8") as f:
            task_hierarchy = json.load(f)
        
        # Traverse the task hierarchy to find the correct location to insert the smaller tasks 
        # (goal and implementation are slightly different from get_curr_subtask function).
        if (debug): print("Traversing task hierarchy to find correct location for smaller tasks...")
        super_task = task_hierarchy
        for t, _ in super_task_prefix[1:]:  # skip the first element of super_task_prefix which corresponds to the original task
            curr_task = super_task["subtasks"][t]
            curr_task_objective = curr_task["task_objective"]
            super_task = curr_task
            if ("subtasks" not in curr_task or len(curr_task["subtasks"]) == 0):
                break
        if (debug): print(f"Found location in task hierarchy for smaller tasks. Super task objective: {super_task['task_objective']}")
        
        # Insert the smaller tasks into the hierarchy
        if ("subtasks" not in super_task.keys()): super_task["subtasks"] = []
        for smaller_task in smaller_tasks_list:
            super_task["subtasks"].append({
                "task_objective": smaller_task,
                "finished": False
            })
        if (debug): print(f"Inserted smaller tasks into hierarchy: {[t['task_objective'] for t in super_task['subtasks']]}")
        
        # Save the updated hierarchy back to the JSON file
        with open(task_hierarchy_path, "w", encoding="utf-8") as f:
            json.dump(task_hierarchy, f, indent=2, ensure_ascii=False)
        if (debug): print("Saved updated task hierarchy to file.")
    
    # TODO: need to test this function
    def analyse_performance(self, max_frames=10, debug=False):
        """
        Analyse the VLA's performance based on the video of its trajectory 
        for the current subtask history step.

        Args:
            max_frames (int): the maximum number of frames to sample from the video.
            debug (bool): whether to print debug information.
        Returns:
            str: the VLM's textual analysis of how the VLA performed for the current step (original prompt removed).
        """
        # Load analysis system prompt
        mm_step_analysis_prompt_path = Path(self.vlm_mm_prompts_dir) / "mm_step_analysis_prompt.txt"
        if not mm_step_analysis_prompt_path.exists():
            raise FileNotFoundError(f"Expected analysis prompt at {mm_step_analysis_prompt_path} but file not found.")
        with open(mm_step_analysis_prompt_path, "r", encoding="utf-8") as f:
            analysis_system_prompt = f.read().strip()

        # Load video from video_path using opencv
        video_path = self.vla_context_video_path
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        frames = []
        if cv2 is not None:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                cap.release()
                raise RuntimeError(f"Failed to open video: {video_path}")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # BGR -> RGB
                frame = frame[:, :, ::-1]
                frames.append(Image.fromarray(frame))
            cap.release()
        else:
            reader = imageio.get_reader(str(video_path))
            try:
                for frame in reader:
                    # imageio returns already RGB
                    frames.append(Image.fromarray(frame))
            finally:
                reader.close()

        if len(frames) == 0:
            raise RuntimeError("No frames extracted from video.")

        # Load mm history and task hierarchy to provide full context for analysis
        mm_history_path = self.vlm_mm_context_dir / "mm_history.json"
        with open(mm_history_path, "r", encoding="utf-8") as f:
            mm_history = json.load(f)

        # Current subtask objective
        curr_subtask = self.get_curr_subtask_mm_history(mm_history)
        curr_subtask_objective = curr_subtask["task_objective"]
        curr_subtask_history = curr_subtask["task_history"]
        
        # Evenly sample video frames to avoid extremely large prompts.
        sampled_frames = None
        if len(frames) > max_frames:
            indices = torch.linspace(0, len(frames) - 1, steps=max_frames, dtype=torch.long)
            sampled_frames = [frames[i] for i in indices]
        else:
            sampled_frames = frames

        # Build analysis prompt including each video frame as an image message (most recent first)
        analysis_prompt = [
            {
            "role": "system",
            "content": [
                {"type": "text", "text": analysis_system_prompt}
            ]
            },
            {
            "role": "user",
            "content": (
                [
                    {"type": "image", "image": img} for img in sampled_frames
                ] + [
                    {"type": "text", "text": f"Current subtask objective: {curr_subtask_objective}"},
                    {"type": "text", "text": "Current subtask history (most recent first):"},
                    {"type": "text", "text": json.dumps(curr_subtask_history, indent=2, ensure_ascii=False)},
                ]
            )
            },
        ]

        # Process inputs and run generation
        inputs = self.processor.apply_chat_template(
            analysis_prompt,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.model.device)

        with torch.inference_mode():
            output = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            # cache_implementation="static",
            )

        # Extract generated token ids (exclude prompt tokens)
        if getattr(self.model.config, "is_encoder_decoder", False):
            generated_ids = output[0]
        else:
            prompt_length = inputs.input_ids.shape[1]
            generated_ids = output[0][prompt_length:]

        generated_text = self.processor.decode(generated_ids, skip_special_tokens=True).strip()

        # Ensure the original system/user prompt text is not part of the returned output
        cleaned = generated_text.replace(analysis_system_prompt, "")
        if hasattr(self, "vla_original_prompt"):
            cleaned = cleaned.replace(self.vla_original_prompt, "")
        cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned).strip()

        # Save analysis to mm_outputs file
        curr_subtask_history[-1]["robot_performance_results"] = cleaned
        with open(mm_history_path, "w", encoding="utf-8") as f:
            json.dump(mm_history, f, indent=2, ensure_ascii=False)
        
        if (debug): print("Saved VLA performance analysis to mm_history.json")

        return cleaned
    
    def _extract_json_from_text(self, text):
        """
        Extract the first JSON object from arbitrary text and return it as a dict.
        Supports fenced ```json blocks and raw {...} with nested braces.
        Attempts strict json.loads first, then a conservative single-quote -> double-quote
        fallback if parsing fails.
        Raises ValueError if no valid JSON object can be parsed.
        """
        # Find fenced ```json blocks
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
        if m:
            candidate = m.group(1).strip()
            return json.loads(candidate)
        else:
            return {}
            
            
    ### OLD VERSIONS OF FUNCTIONS - can be deleted after new versions are verified to work correctly ###
    def get_curr_task_OLD(self):
        return self.task_plan[self.curr_task]

    def update_curr_task_OLD(self, prev_subtask_status, debug=False):
        if (self.curr_task == len(self.task_plan) - 1):
            if (debug):
                print("Already at the last task in the plan.")
            return self.curr_task
        
        if (prev_subtask_status == "completed"):
            self.curr_task += 1
            if (debug):
                print(f"Moving to next task: {self.get_curr_task()}")
        else:
            if (debug):
                print(f"Previous subtask status: {prev_subtask_status}. Staying on current task: {self.get_curr_task()}")
    
    def get_curr_subtask_OLD(self, mm_history):
        
        def r_get_latest_subtask(curr_task):
            curr_subtask = None
            for t in curr_task["task_history"][::-1]:
                if t["mm_output_type"] == "smaller_tasks":
                    curr_subtask = t["subtasks"][-1]
                    return r_get_latest_subtask(curr_subtask)
            return curr_task
        
        return r_get_latest_subtask(mm_history[-1])

    def get_mm_curr_step_inputs_OLD(self, history_window=20):
        """
        Get the inputs for the VLM to determine the current step. This will include:
            - From VLA History [k-n, k-1]:
                - Step #
                - Subtask objective
                - Input prompt (can be a revised version of the subtask objective)
                - Input image
                - Action used
                - Action source
            - From MM History [k-n, k-1]:
                - Corresponding VLA history step #
                - Previous subtask status
                - Revised prompt
                - VLA's subtask trajectory explanation
                - MM's subtask output explanation
            - Current (k_th) subtask prompt
            - Current (k_th) subtask input image

        Returns:
            dict: containing vla_history, mm_history, current_subtask_prompt,
                  current_subtask_input_image
        """
        # Step 1: Load VLA history and MM history from files
        vla_history_path = self.vlm_mm_context_dir / "vla_history.json"
        mm_history_path = self.vlm_mm_context_dir / "mm_history.json"

        with open(vla_history_path, "r", encoding="utf-8") as f:
            vla_history = json.load(f)
        
        with open(mm_history_path, "r", encoding="utf-8") as f:
            mm_history = json.load(f)
        
        # Step 2: Retrieve necessary information from VLA history
        vla_t_history = []
        for history_step in vla_history[max(-history_window, 0):]:
            vla_t_history.append({
                "step_num": history_step["step_num"],
                "subtask_objective": history_step["subtask_objective"],
                "input_prompt": history_step["input_prompt"],
                "input_image_path": history_step["input_image_path"],
                "action_used": history_step["vla_action"] if history_step["chosen_action"] == "vla" else history_step["mm_action"],
                "action_source": history_step["chosen_action"],
            })
        
        # Step 3: Retrieve necessary information from MM history
        mm_t_history = []
        for history_step in mm_history[max(-history_window, 0):]:
            if (history_step["vla_history_step_num"] < vla_t_history[0]["step_num"]):
                continue
            mm_t_history.append({
                "vla_history_step_num": history_step["vla_history_step_num"],
                "previous_subtask_status": history_step["previous_subtask_status"],
                "revised_prompt": history_step["revised_prompt"],
                "override_action": history_step["mm_override_action"],
                "vla_subtask_trajectory_explanation": history_step["vla_subtask_trajectory_explanation"],
                "mm_subtask_output_explanation": history_step["mm_subtask_output_explanation"],
            })
        
        # Step 4: Get current subtask objective
        self.curr_task = vla_history[-1]["subtask_idx"]
        prev_subtask_status = mm_history[-1]["previous_subtask_status"]
        self.update_curr_task(prev_subtask_status)
        current_subtask_objective = self.get_curr_task()

        # Step 5: Get current subtask input image
        current_subtask_input_image = self.vla_curr_image_path # get this image in file so that it does not get overwritten and cause race conditions?

        return {
            "vla_history": vla_t_history,
            "mm_history": mm_t_history,
            "current_subtask_objective": current_subtask_objective,
            "current_subtask_input_image": current_subtask_input_image,
        }

    def save_mm_curr_step_outputs_OLD(self, curr_step, curr_mm_outputs):
        """
        Save the VLM MM's outputs for the current step. This will include:
            - Corresponding VLA history step number
            - Previous subtask status (e.g. "completed", "in_progress", "not_started")
            - Override action (if any)
            - Reasoning behind VLA trajectory movement for the current subtask
            - Reasoning behind MM's outputs
        """
        mm_outputs_path = self.vlm_mm_context_dir / "mm_outputs.json"

        curr_mm_outputs["vla_history_step_num"] = curr_step
        assert list(curr_mm_outputs.keys()) == [
            "vla_history_step_num", "previous_subtask_status", "revised_prompt", "mm_override_action", "vla_reasoning", "mm_reasoning"
        ], "curr_mm_outputs must contain keys: previous_subtask_status, revised_prompt, mm_override_action, vla_reasoning, mm_reasoning"

        with open(mm_outputs_path, "r", encoding="utf-8") as f:
            mm_outputs = json.load(f)
        
        mm_outputs.append(curr_mm_outputs)
        with open(mm_outputs_path, "w", encoding="utf-8") as f:
            json.dump(mm_outputs, f, indent=2, ensure_ascii=False)



if __name__ == "__main__":
    vlm_mm = VLMMM(
        model_name="gemma3",
        vlm_mm_context_dir="vlm_mm/mm_context",
        vlm_mm_prompts_dir="vlm_mm/mm_prompts",
        vla_original_prompt="put the black bowl in the bottom drawer of the cabinet and close it",
        vla_curr_image_path="/home/hice1/apalasamudram6/scratch/vlm_mm_vla/vlm_mm/mm_context/curr_image.png",
    )

    vlm_mm.step(debug=True)
    vlm_mm.step(debug=True)

    # task_plan = vlm_mm.create_task_plan(debug=True)

    # task_hierarchy_path = vlm_mm.vlm_mm_context_dir / "task_hierarchy.json"
    # with open(task_hierarchy_path, "r", encoding="utf-8") as f:
    #     task_hierarchy = json.load(f)
    # vlm_mm.curr_task_prefix = [(0,1), (2,2)]
    # vlm_mm.update_task_hierarchy(vlm_mm.curr_task_prefix, ["task a", "task b", "task c"])
    # next_task_prefix = vlm_mm.get_next_step_task_prefix(vlm_mm.curr_task_prefix, task_hierarchy, new_subtasks_created=False, debug=False)
    # print("Next task prefix:", next_task_prefix)
    # print(vlm_mm.get_subtask(vlm_mm.curr_task_prefix, task_hierarchy)["task_objective"])
    # mm_task = vlm_mm.get_curr_subtask_mm_history()
    # print("Current MM task:", mm_task["task_objective"])
    # print("Current MM task history index:", mm_task["task_history"][vlm_mm.curr_task_prefix[-1][1]])
    
    # while (vlm_mm.curr_task_prefix is not None):
    #     next_task_prefix = vlm_mm.get_next_step_task_prefix(vlm_mm.curr_task_prefix, task_hierarchy)
    #     print("Next task prefix:", next_task_prefix)
    #     print(vlm_mm.get_subtask(next_task_prefix, task_hierarchy)["task_objective"])
    #     vlm_mm.curr_task_prefix = next_task_prefix
