import os
import json
import argparse
import datetime
import cv2
import numpy as np

from utils import *
from vlm import ModelHandler
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Predict and Evaluate Custom Dataset")
    parser.add_argument("--model", type=str, default="qwen2.5-72b")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--videos_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--error_handling", action="store_true")
    return parser.parse_args()

def calculate_custom_metrics(pred_traj, gt_traj):
    pred, gt = np.array(pred_traj), np.array(gt_traj)
    min_len = min(len(pred), len(gt))
    if min_len == 0: return None
    
    pred, gt = pred[:min_len], gt[:min_len]
    dist = np.linalg.norm(pred - gt, axis=1)
    
    return {
        "ADE_1s": np.mean(dist[:2]).item() if min_len >= 2 else None,
        "ADE_2s": np.mean(dist[2:4]).item() if min_len >= 4 else None,
        "ADE_3s": np.mean(dist[4:6]).item() if min_len >= 6 else None,
        "ADE_avg": np.mean(dist).item(),
        "FDE": dist[-1].item(),
        "missRate_2": 1.0 if np.max(dist) > 2.0 else 0.0
    }

def run_pipeline():
    args = parse_args()
    config = load_config("config.yaml")
    
    model_handler = ModelHandler(args.model, config)
    model_handler.initialize_model()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = os.path.join(args.output_dir, f"{args.model}_{timestamp}")
    frames_out_dir = os.path.join(results_dir, "extracted_frames")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(frames_out_dir, exist_ok=True)
    
    videos = [v for v in sorted(os.listdir(args.videos_dir)) if v.endswith('.mp4')]
    if args.scene: videos = [v for v in videos if args.scene in v]
    
    global_metrics = {"ADE_avg": [], "FDE": [], "missRate_2": [], "frames_total": 0, "frames_success": 0}
    
    for video_file in videos:
        scene_name = video_file.split('.')[0]
        scene_meas_dir = os.path.join(args.dataset_dir, scene_name, "measurements")
        
        if not os.path.exists(scene_meas_dir): raise FileNotFoundError(f"Measurements directory not found for scene: {scene_name}")
            
        json_files = sorted([f for f in os.listdir(scene_meas_dir) if f.endswith('.json')])
        measurements = [json.load(open(os.path.join(scene_meas_dir, f))) for f in json_files]
        if len(measurements) < 109: raise ValueError(f"Not enough measurement files for scene: {scene_name}. Found {len(measurements)}, expected at least 109.")
            
        print(f"\nProcessing & Evaluating Scene: {scene_name}")
        
        cap = cv2.VideoCapture(os.path.join(args.videos_dir, video_file))
        extracted_frames = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            path = os.path.join(frames_out_dir, f"{scene_name}_{frame_idx:03d}.jpg")
            cv2.imwrite(path, frame)
            extracted_frames.append(path)
            frame_idx += 1
        cap.release()
        
        scene_data = {"scene_info": {"name": scene_name}, "frames": [], "scene_metrics": {}}
        scene_ades, scene_fdes, scene_misses = [], [], []
        
        for i, image_path in enumerate(tqdm(extracted_frames, desc=f"Frames for {scene_name}")):
            if i % 2 != 0: continue  # Process every 2nd frame, i.e. 5 Hz
            cur_idx = i + 30 
            cur_meas = measurements[cur_idx]
            cur_pos = (cur_meas["pos_global"][0], -cur_meas["pos_global"][1])
            cur_heading = -cur_meas["theta"]
            
            past_idx = range(cur_idx - 25, cur_idx + 1, 5)
            fut_idx = range(cur_idx + 5, cur_idx + 31, 5) # Corrected range
            
            prev_speed = [measurements[idx]["speed"] for idx in past_idx]
            
            obs_pos_raw = [measurements[idx]["pos_global"] for idx in range(cur_idx - 30, cur_idx + 1)]
            obs_pos = [(p[0], -p[1]) for p in obs_pos_raw]
            obs_pos_ego = global_to_ego_frame(cur_pos, cur_heading, obs_pos)
            prev_curvatures = compute_curvature(obs_pos_ego)[::5][-6:]
            
            fut_pos_raw = [measurements[idx]["pos_global"] for idx in fut_idx]
            fut_pos = [(p[0], -p[1]) for p in fut_pos_raw]
            gt_positions = global_to_ego_frame(cur_pos, cur_heading, fut_pos)
            
            # Restored exact prompts
            scene_prompt = (
                "You are an autonomous driving labeller. "
                "You have access to the front-view camera image. "
                "You must observe and analyze the movements of vehicles and pedestrians, "
                "lane markings, traffic lights, and any relevant objects in the scene. "
                "describe what you observe, but do not infer the ego's action. "
                "generate your response in plain text in one paragraph without any formating. "
            )
            scene_description, _, _ = model_handler.get_response(scene_prompt, image_path)
            print(f"\nScene Description for frame {i}:", scene_description)
            print(f"Previous Speed for frame {i}:", prev_speed)
            print(f"Previous Curvatures for frame {i}:", prev_curvatures)

            intent_prompt = (
                "You are an autonomous driving labeller. "
                "You have access to the front-view camera image. "
                "The scene is described as follows: "
                f"{scene_description} "
                "The ego vehicle's speed for the past 3 seconds with 0.5 sec resolution is"
                f"{prev_speed} m/s (last index is the most recent) "
                "The ego vehicle's curvature for the past 3 seconds with 0.5 sec resolution is"
                f"{prev_curvatures} (last index is the most recent) "
                "A positive curvature indicates the ego is turning left."
                "A negative curvature indicates the ego is turning right. "
                "What was the ego's previous intent? "
                "Was it accelerating (by how much), decelerating (by how much), or maintaining speed? "
                "Was it turning left (by how much), turning right (by how much), or following the lane? "
                "Taking into account the ego's previous intent, how should it drive in the next 3 seconds? "
                "Should the ego accelerate (by how much), decelerate (by how much), or maintain speed? "
                "Should the ego turn left (by how much), turn right (by how much), or follow the lane?  "
                "Generate your response in plain text in one paragraph without any formating. "
            )
            driving_intent, _, _ = model_handler.get_response(intent_prompt, image_path)
            print(f"\nDriving Intent for frame {i}:", driving_intent)

            waypoint_prompt = (
                "You are an autonomous driving labeller. "
                "You have access to the front-view camera image. "
                "The scene is described as follows: "
                f"{scene_description} "
                "The ego vehicle's speed for the past 3 seconds with 0.5 sec resolution is"
                f"{prev_speed} m/s (last index is the most recent) "
                "The ego vehicle's curvature for the past 3 seconds with 0.5 sec resolution is"
                f"{prev_curvatures} (last index is the most recent) "
                "A positive curvature indicates the ego is turning left."
                "A negative curvature indicates the ego is turning right. "
                "The high-level driving instructions are as follows: "
                f"{driving_intent} "
                "Predict the speed and curvature for the next 6 waypoints, with 0.5-second resolution. "
                "The predicted speed and curvature changes must obey the physical constraints of the vehicle. "
                "Predict Exactly 6 pairs of speed and curvature, in the format:"
                "[(v1, c1), (v2, c2), (v3, c3), (v4, c4), (v5, c5), (v6, c6)]. "
                "ONLY return the answers in the required format, do not include punctuation or text."
            )
            actions_str, _, _ = model_handler.get_response(waypoint_prompt, image_path)
            print(f"\nPredicted Actions for frame {i}:", actions_str)
            print("=" * 50)
            # Evaluation
            frame_metrics = None
            pred_actions = extract_driving_action(actions_str, args.error_handling)
            global_metrics["frames_total"] += 1
            
            if pred_actions:
                pred_traj = integrate_driving_commands(pred_actions, dt=0.5)
                frame_metrics = calculate_custom_metrics(pred_traj, gt_positions)
                if frame_metrics:
                    scene_ades.append(frame_metrics["ADE_avg"])
                    scene_fdes.append(frame_metrics["FDE"])
                    scene_misses.append(frame_metrics["missRate_2"])
                    global_metrics["ADE_avg"].append(frame_metrics["ADE_avg"])
                    global_metrics["FDE"].append(frame_metrics["FDE"])
                    global_metrics["missRate_2"].append(frame_metrics["missRate_2"])
                    global_metrics["frames_success"] += 1

            scene_data["frames"].append({
                "frame_index": i,
                "image_name": os.path.basename(image_path),
                "metrics": frame_metrics,
                "inference": {"actions": actions_str}
            })
            
        # Scene Averages
        if scene_ades:
            scene_data["scene_metrics"] = {
                "ADE_avg": np.mean(scene_ades).item(),
                "FDE": np.mean(scene_fdes).item(),
                "missRate_2": np.mean(scene_misses).item()
            }
            
        save_dict_to_json(scene_data, os.path.join(results_dir, f"{scene_name}.json"))

        for frame_path in extracted_frames:
            if os.path.exists(frame_path):
                os.remove(frame_path)
        print(f"Deleted {len(extracted_frames)} temporary frames for {scene_name}.")
    
    # Global Averages
    if global_metrics["frames_success"] > 0:
        final_log = {
            "Total_Frames": global_metrics["frames_total"],
            "Successful_Frames": global_metrics["frames_success"],
            "Success_Rate": global_metrics["frames_success"] / global_metrics["frames_total"],
            "Global_ADE_avg": np.mean(global_metrics["ADE_avg"]).item(),
            "Global_FDE": np.mean(global_metrics["FDE"]).item(),
            "Global_missRate_2": np.mean(global_metrics["missRate_2"]).item()
        }
        save_dict_to_json(final_log, os.path.join(results_dir, "evaluation_log.json"))
        print("\nGlobal Evaluation Log Saved:", final_log)

if __name__ == "__main__":
    run_pipeline()