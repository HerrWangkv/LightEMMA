import os
import cv2
import json
import argparse
import shutil
import subprocess
import numpy as np
from tqdm import tqdm

from pyquaternion import Quaternion

from utils import (
    extract_driving_action,
    integrate_driving_commands,
    global_to_ego_frame
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_dir", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output")
    return parser.parse_args()


# ------------------------------------------------------------------
# Camera calibration loading (correct)
# ------------------------------------------------------------------
def load_camera_params(dataset_dir):

    scenes = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])

    cam_json_path = os.path.join(
        dataset_dir,
        scenes[0],
        "calibrated_sensor.json"
    )

    with open(cam_json_path, 'r') as f:
        cam = json.load(f)["CAM_FRONT"]

    intrinsic = np.array(cam["intrinsic"], dtype=float)

    extr = cam["extrinsic"]

    loc = extr["location"]
    rot = extr["rotation"]

    # Convert quaternion → rotation matrix
    q = Quaternion(
        rot["w"],
        rot["x"],
        rot["y"],
        rot["z"]
    )

    R = q.rotation_matrix

    t = np.array([
        loc["x"],
        loc["y"],
        loc["z"]
    ], dtype=float)

    # ego → camera transform
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R.T
    extrinsic[:3, 3] = -R.T @ t

    return intrinsic, extrinsic


# ------------------------------------------------------------------
# Load GT trajectory correctly (same logic as eval script)
# ------------------------------------------------------------------
def load_gt_trajectory(dataset_dir, scene_id, frame_idx):

    meas_dir = os.path.join(dataset_dir, scene_id, "measurements")

    json_files = sorted([
        f for f in os.listdir(meas_dir)
        if f.endswith(".json")
    ])

    measurements = [
        json.load(open(os.path.join(meas_dir, f)))
        for f in json_files
    ]

    cur_idx = frame_idx + 30

    if cur_idx + 30 >= len(measurements):
        return None

    cur_meas = measurements[cur_idx]

    cur_pos = (
        cur_meas["pos_global"][0],
        -cur_meas["pos_global"][1]
    )

    cur_heading = -cur_meas["theta"]

    fut_idx = range(cur_idx + 5, cur_idx + 31, 5)

    fut_pos_raw = [
        measurements[idx]["pos_global"]
        for idx in fut_idx
    ]

    fut_pos = [
        (p[0], -p[1])
        for p in fut_pos_raw
    ]

    gt_traj = global_to_ego_frame(
        cur_pos,
        cur_heading,
        fut_pos
    )

    return np.array(gt_traj)


# ------------------------------------------------------------------
# Convert predicted actions to trajectory
# ------------------------------------------------------------------
def load_pred_trajectory(actions_str):

    actions = extract_driving_action(
        actions_str,
        error_handling=True
    )

    if actions is None:
        return None

    traj = integrate_driving_commands(actions, dt=0.5)

    return np.array(traj)


# ------------------------------------------------------------------
# Projection
# ------------------------------------------------------------------
def project_trajectory(frame, trajectory, intrinsic, extrinsic, color):

    if trajectory is None or len(trajectory) == 0:
        return frame
    trajectory = trajectory.copy()

    FRONT_OFFSET = 4  # meters (adjust as needed)
    trajectory[:,0] += FRONT_OFFSET
    if trajectory.shape[1] == 2:
        trajectory = np.hstack([
            trajectory,
            np.zeros((trajectory.shape[0], 1))
        ])

    ones = np.ones((trajectory.shape[0], 1))

    pts = np.hstack([trajectory, ones])

    pts_cam = (extrinsic @ pts.T).T

    valid = pts_cam[:, 2] > 0

    pts_cam = pts_cam[valid]

    proj = (intrinsic @ pts_cam[:, :3].T).T

    uv = proj[:, :2] / proj[:, 2][:, None]

    uv = uv.astype(int)

    for i in range(len(uv) - 1):

        overlay = frame.copy()
        cv2.line(
            overlay,
            tuple(uv[i]),
            tuple(uv[i+1]),
            color,
            3
        )
        alpha = 0.5  # transparency factor
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        cv2.circle(frame, tuple(uv[i]), 5, color, -1)

    if len(uv) > 0:
        cv2.circle(frame, tuple(uv[-1]), 5, color, -1)

    return frame


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():

    args = parse_args()

    intrinsic, extrinsic = load_camera_params(args.dataset_dir)

    scale_x = 1280 / 1600
    scale_y = 704 / 900

    intrinsic_scaled = intrinsic.copy()

    intrinsic_scaled[0, 0] *= scale_x
    intrinsic_scaled[0, 2] *= scale_x
    intrinsic_scaled[1, 1] *= scale_y
    intrinsic_scaled[1, 2] *= scale_y

    results_dir = os.path.join(args.output_dir, args.exp_name)

    out_dir = os.path.join(results_dir, "reprojected_videos")

    os.makedirs(out_dir, exist_ok=True)

    json_files = [
        f for f in os.listdir(results_dir)
        if f.endswith(".json")
        and f != "evaluation_log.json"
    ]

    for json_file in tqdm(json_files):

        scene_id = json_file.replace(".json", "")

        video_path = os.path.join(
            args.videos_dir,
            f"{scene_id}.mp4"
        )

        if not os.path.exists(video_path):
            continue

        scene_data = json.load(
            open(os.path.join(results_dir, json_file))
        )

        logs = scene_data["frames"]

        cap = cv2.VideoCapture(video_path)

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 10

        temp_dir = os.path.join(out_dir, f"temp_{scene_id}")
        os.makedirs(temp_dir, exist_ok=True)

        frame_idx = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.resize(frame, (1280, 704))
            logs = scene_data["frames"]

            log_dict = {
                entry["frame_index"]: entry
                for entry in logs
            }
            if frame_idx in log_dict:

                log = log_dict[frame_idx]

                if log["metrics"] is not None:

                    gt = load_gt_trajectory(
                        args.dataset_dir,
                        scene_id,
                        frame_idx
                    )

                    pred = load_pred_trajectory(
                        log["inference"]["actions"]
                    )

                    frame = project_trajectory(
                        frame,
                        gt,
                        intrinsic_scaled,
                        extrinsic,
                        (0,255,0)
                    )

                    frame = project_trajectory(
                        frame,
                        pred,
                        intrinsic_scaled,
                        extrinsic,
                        (0,0,255)
                    )

            cv2.imwrite(
                os.path.join(
                    temp_dir,
                    f"{frame_idx:04d}.png"
                ),
                frame
            )

            frame_idx += 1

        cap.release()

        out_path = os.path.join(
            out_dir,
            f"{scene_id}_reprojected.mp4"
        )

        subprocess.run([
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            os.path.join(temp_dir, "%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            out_path
        ])

        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()