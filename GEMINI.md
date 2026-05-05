# LightEMMA: Project Instructions & Context

This document provides essential context and instructions for working with the **LightEMMA** project.

## Project Overview

**LightEMMA** is a lightweight, modular framework for end-to-end autonomous driving research. It leverages the zero-shot reasoning capabilities of Vision-Language Models (VLMs) to predict driving actions and trajectories from front-camera images.

### Architecture & Pipeline
The framework follows a three-step **Chain-of-Thought (CoT)** reasoning approach:
1.  **Scene Description**: Generates a detailed description of the environment (lanes, traffic lights, objects).
2.  **Driving Intent Analysis**: Predicts high-level maneuvers (e.g., "accelerate and turn left") based on the scene and historical ego-state.
3.  **Trajectory Prediction**: Converts intent into six waypoints (0.5s resolution) consisting of speed and curvature.

### Key Technologies
- **Language**: Python 3.10
- **VLMs**: GPT (OpenAI), Claude (Anthropic), Gemini (Google), Qwen (Local), LLaMA (Local).
- **Dataset**: [nuScenes](https://www.nuscenes.org/nuscenes) (prediction benchmark).
- **Geometry**: \`pyquaternion\`, \`shapely\`, \`nuscenes-devkit\`.

## Building and Running

### Environment Setup
1.  **Conda Environment**:
    \`\`\`bash
    conda create -n lightemma python=3.10
    conda activate lightemma
    pip install -r requirements.txt
    \`\`\`
2.  **Configuration**: Update \`config.yaml\` with API keys, Hugging Face tokens, and the local path to the nuScenes dataset.

### Core Commands
- **Prediction**:
  \`\`\`bash
  python predict.py --model chatgpt-4o-latest
  # For specific scene:
  python predict.py --model chatgpt-4o-latest --scene scene-0103
  \`\`\`
- **Evaluation**:
  \`\`\`bash
  python evaluate.py --results_dir results/gpt-4o
  # With error handling and visualization:
  python evaluate.py --results_dir results/gpt-4o --error_handling --visualize
  \`\`\`
- **Batch Evaluation**:
  \`\`\`bash
  python evaluate_all.py --results_dir results
  \`\`\`
- **CARLA / Custom Dataset Pipeline**:
  \`\`\`bash
  python predict_and_eval_carla.py \
    --model gemini-2.5-flash \
    --dataset_dir path/to/dataset \
    --videos_dir path/to/videos \
    --exp_name my_experiment
  \`\`\`
- **Visualization (Reprojection)**:
  \`\`\`bash
  python plot_trajectories.py \
    --videos_dir path/to/videos \
    --dataset_dir path/to/dataset \
    --exp_name my_experiment
  \`\`\`

## Development Conventions

### Code Structure
- \`vlm.py\`: Contains \`ModelHandler\` for abstracting interactions with various VLM backends (API vs. Local).
- \`predict.py\`: Main entry point for the nuScenes prediction pipeline.
- \`predict_and_eval_carla.py\`: Unified pipeline for CARLA or custom datasets; handles frame extraction from videos and simultaneous evaluation.
- \`plot_trajectories.py\`: Visualization tool that reprojects 2D ego-frame waypoints onto the camera image plane to generate videos with overlaid paths.
- \`evaluate.py\`: Logic for calculating ADE (Average Displacement Error), FDE (Final Displacement Error), and Miss Rate.
- \`utils.py\`: Geometric transformations (global to ego frame), trajectory integration, and visualization (overlaying paths on images).

### Data Handling
- **Datasets**: 
    - **nuScenes**: Standard autonomous driving dataset; processed via \`predict.py\`.
    - **CARLA / Custom**: Processed via \`predict_and_eval_carla.py\`. Requires video files (\`.mp4\`) and corresponding measurement JSON files. Frames are extracted on-the-fly at 5 Hz.
- **Results**: Stored in JSON format. Each scene has its own JSON file containing frames, prompts, model responses, and metadata.
- **Reprojection**: The process of projecting 3D waypoints (from the ego frame) onto the 2D image plane.
    - **Green Path**: Ground Truth trajectory.
    - **Red Path**: Predicted trajectory.
    - Requires camera intrinsics and extrinsics (loaded from \`calibrated_sensor.json\`).
- **Coordinate Frames**: The project frequently transforms between **Global** (nuScenes/World) and **Ego** (vehicle-relative) frames. Note that CARLA uses a left-handed coordinate system, so the script negates the Y-coordinate and heading during transformation.

- **Inference Persistence**: Use the \`--continue_dir\` flag in \`predict.py\` to resume interrupted runs from a results directory.

### Testing & Validation
- Currently, the project lacks traditional unit tests; validation is primarily performed through the evaluation scripts (\`evaluate.py\`) on the nuScenes dataset.
- When adding new models or features, verify performance changes using the \`ADE\` and \`FDE\` metrics.

### Styling & Standards
- **CLI**: Standard \`argparse\` for scripts.
- **Config**: Centralized in \`config.yaml\`.
- **Naming**: Snake_case for variables and functions; CamelCase for classes (e.g., \`ModelHandler\`).
- **Formatting**: JSON results use structured lists for long text to maintain readability.
