import cv2
import numpy as np
from collections import deque
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

class MotionSurpriseScorer:
    """
    CPU-friendly motion-based surprise scoring methods for video analysis.
    """
    
    def __init__(self, window_size: int = 10, background_learning_rate: float = 0.01):
        self.window_size = window_size
        self.background_learning_rate = background_learning_rate
        
        # For tracking motion history
        self.motion_history = deque(maxlen=window_size)
        self.frame_history = deque(maxlen=2)  # Only need current and previous
        
        # For background subtraction
        self.background_model = None
        self.background_initialized = False
        
        # For optical flow
        self.prev_gray = None
        self.lk_params = dict(winSize=(15, 15),
                             maxLevel=2,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame to grayscale and apply slight blur for noise reduction."""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame.copy()
        return cv2.GaussianBlur(gray, (5, 5), 0)
    
    def method_1_frame_difference(self, frame: np.ndarray) -> float:
        """
        Method 1: Simple frame difference approach
        Surprise = magnitude of change between consecutive frames
        """
        processed_frame = self.preprocess_frame(frame)
        self.frame_history.append(processed_frame)
        
        if len(self.frame_history) < 2:
            return 0.0
        
        # Calculate absolute difference
        diff = cv2.absdiff(self.frame_history[-2], self.frame_history[-1])
        
        # Calculate surprise score (normalized mean absolute difference)
        surprise_score = np.mean(diff) / 255.0
        
        return surprise_score
    
    def method_2_optical_flow_magnitude(self, frame: np.ndarray) -> float:
        """
        Method 2: Optical flow magnitude
        Uses Farneback optical flow to measure motion intensity
        """
        processed_frame = self.preprocess_frame(frame)
        
        if self.prev_gray is None:
            self.prev_gray = processed_frame
            return 0.0
        
        # Calculate optical flow using Farneback method
        flow = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, processed_frame, None, None, **self.lk_params
        )
        
        # For Farneback method (denser flow):
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, processed_frame, None, 
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Calculate magnitude of flow vectors
        magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        
        # Surprise score based on mean flow magnitude
        surprise_score = np.mean(magnitude)
        
        self.prev_gray = processed_frame
        return surprise_score
    
    def method_3_motion_energy(self, frame: np.ndarray) -> float:
        """
        Method 3: Motion energy based on temporal gradients
        Measures energy in temporal changes across the frame
        """
        processed_frame = self.preprocess_frame(frame)
        self.frame_history.append(processed_frame)
        
        if len(self.frame_history) < 2:
            return 0.0
        
        # Calculate temporal gradient
        temporal_grad = self.frame_history[-1].astype(float) - self.frame_history[-2].astype(float)
        
        # Calculate spatial gradients
        grad_x = cv2.Sobel(processed_frame, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(processed_frame, cv2.CV_64F, 0, 1, ksize=3)
        
        # Motion energy combines temporal and spatial information
        motion_energy = np.sqrt(temporal_grad**2 + 0.1 * (grad_x**2 + grad_y**2))
        
        # Surprise score
        surprise_score = np.mean(motion_energy) / 255.0
        
        return surprise_score
    
    def method_4_background_subtraction(self, frame: np.ndarray) -> float:
        """
        Method 4: Background subtraction approach
        Surprise based on deviation from learned background model
        """
        processed_frame = self.preprocess_frame(frame).astype(float)
        
        if not self.background_initialized:
            self.background_model = processed_frame.copy()
            self.background_initialized = True
            return 0.0
        
        # Update background model with exponential averaging
        self.background_model = ((1 - self.background_learning_rate) * self.background_model + 
                                self.background_learning_rate * processed_frame)
        
        # Calculate foreground (surprise) as difference from background
        foreground = np.abs(processed_frame - self.background_model)
        
        # Apply threshold to reduce noise
        threshold = 30
        foreground_mask = foreground > threshold
        
        # Surprise score based on amount and intensity of foreground
        surprise_score = (np.sum(foreground * foreground_mask) / 
                         (np.sum(foreground_mask) + 1e-6)) / 255.0
        
        return surprise_score
    
    def method_5_adaptive_motion_baseline(self, frame: np.ndarray) -> float:
        """
        Method 5: Adaptive baseline that combines multiple motion cues
        Uses running statistics to normalize surprise scores
        """
        # Get scores from multiple methods
        frame_diff_score = self.method_1_frame_difference(frame)
        
        # Store in motion history for adaptive thresholding
        self.motion_history.append(frame_diff_score)
        
        if len(self.motion_history) < self.window_size:
            return frame_diff_score
        
        # Calculate adaptive statistics
        motion_array = np.array(self.motion_history)
        motion_mean = np.mean(motion_array)
        motion_std = np.std(motion_array) + 1e-6
        
        # Z-score based surprise (how unusual is current motion?)
        surprise_score = (frame_diff_score - motion_mean) / motion_std
        
        # Convert to positive score and normalize
        surprise_score = max(0, surprise_score) / 3.0  # Normalize assuming 3-sigma rule
        
        return surprise_score

def process_video_frames(frames: List[np.ndarray], method: str = "frame_difference") -> List[float]:
    """
    Process a list of video frames and return surprise scores.
    
    Args:
        frames: List of video frames as numpy arrays
        method: Which method to use ("frame_difference", "optical_flow", 
                "motion_energy", "background_subtraction", "adaptive")
    
    Returns:
        List of surprise scores for each frame
    """
    scorer = MotionSurpriseScorer(window_size=15)
    scores = []
    
    method_map = {
        "frame_difference": scorer.method_1_frame_difference,
        "optical_flow": scorer.method_2_optical_flow_magnitude,
        "motion_energy": scorer.method_3_motion_energy,
        "background_subtraction": scorer.method_4_background_subtraction,
        "adaptive": scorer.method_5_adaptive_motion_baseline
    }
    
    if method not in method_map:
        raise ValueError(f"Unknown method: {method}")
    
    score_function = method_map[method]
    
    for frame in frames:
        score = score_function(frame)
        scores.append(score)
    
    return scores

def flow_scorer(frames, method: str = "frame_difference") -> List[float]:
    """
    Process a list of video frames and return surprise scores.
    
    Args:
        frames: List of video frames as numpy arrays
        method: Which method to use ("frame_difference", "optical_flow", 
                "motion_energy", "background_subtraction", "adaptive")
    
    Returns:
        List of surprise scores for each frame
    """
    # convert frames to list of np arrays if from tensor
    frames = [frame.numpy() for frame in frames]
    scorer = MotionSurpriseScorer(window_size=4)
    scores = []
    
    method_map = {
        "frame_difference": scorer.method_1_frame_difference,
        "optical_flow": scorer.method_2_optical_flow_magnitude,
        "motion_energy": scorer.method_3_motion_energy,
        "background_subtraction": scorer.method_4_background_subtraction,
        "adaptive": scorer.method_5_adaptive_motion_baseline
    }
    
    if method not in method_map:
        raise ValueError(f"Unknown method: {method}")
    
    score_function = method_map[method]
    
    for frame in frames:
        score = score_function(frame)
        scores.append(score)
    
    return {"surprise_scores": scores}


def evaluate_oops_flow(
    json_path: str,
    video_root: str,
    result_folder: str,
    num_sampled_frames=20,
    num_entries=1,
    model: str = "qwen",
):
    """
    Evaluates all H1-type entries in the FunQA dataset using a surprise scoring method.
    Saves annotated videos with highlighted amusing frames and surprise score bars.
    """

   
    model, processor = load_model_and_processor("Qwen/Qwen2.5-VL-7B-Instruct")

    # recursively make result folder
    if not os.path.exists(result_folder):
        print(f"Creating result folder: {result_folder}")
        os.makedirs(result_folder, exist_ok=True)
    else:
        print(f"Result folder already exists: {result_folder}")

    dataset = load_oops_json(json_path)
    outputs = {}
    random.seed(50)  # For reproducibility

    # sample num_entries from the dataset
    if num_entries is not None:
        dataset = random.sample(dataset, num_entries)
        print(f"Sampling {num_entries} entries from the dataset.")

    # collect percentage of amusing frames
    amusing_fractions = []

    for idx, entry in enumerate(tqdm(dataset, desc="Processing entries")):
        video_file = get_video_path(video_root, entry["set_id"], entry["index"])
        print(f"Processing video {idx+1}/{len(dataset)}: {video_file}")
        if not os.path.exists(video_file):
            print(f"Missing video: {video_file}")
            continue

        #try:
        if video_file in outputs:
            print(f"Already processed: {video_file}")
            continue

        frames, frame_indices, total_frames, fps, vr = extract_k_frames_decord_cpu(
            video_path=video_file, k=num_sampled_frames
        )
        amusing_ids = get_amusing_frame_indices(total_frames, fps, entry["transition"])
        amusing_sampled = [
            int(fid) for fid in frame_indices if int(fid) in amusing_ids
        ]
        frac = len(amusing_sampled) / num_sampled_frames
        amusing_fractions.append(frac)
        frac = len(amusing_ids) / total_frames
 
        if not (0.05 <= frac <= 0.75):
            continue

        # Run your surprise scoring method (e.g. prior_hypothesis_topk_approach)

        output = flow_scorer(
            video_frames=frames, model=model, processor=processor)
        # Replace None with 0.0
        scores_t = torch.tensor(output["surprise_scores"], dtype=torch.float32, device="cpu")
        scores_np = np.array([s.item() for s in scores_t], dtype=float)
        
        output["surprise_scores"] = [
            0.0 if score is None else score for score in scores_np
        ]

        precision, recall, hit = precision_recall_hit_at_k(
            output["surprise_scores"], frame_indices, amusing_sampled, k=5
        )
        prec = round(precision, 2)
        recall = round(recall, 2)
        hit = round(hit, 2)
        outputs[video_file] = {
            "video_path": video_file,
            "amusing_ids": amusing_sampled,
            "surprise_scores": output["surprise_scores"],
            "precision_at_k": precision,
            "recall_at_k": recall,
            "hit_at_1": hit,
            "frame_indices": frame_indices.tolist()
            
        }

        # Convert the single dict to a one-row DataFrame
        row_df = pd.DataFrame([outputs[video_file]])

        result_json_path = os.path.join(result_folder, "results.json")

        # Append to JSON file (newline-delimited JSON)
        if os.path.exists(result_json_path):
            row_df.to_json(result_json_path, orient="records", lines=True, mode="a")
        else:
            row_df.to_json(result_json_path, orient="records", lines=True)

        print(f"Appended results to {result_json_path}")
    # dump final results
    with open(os.path.join(result_folder, "results_final.json"), "w") as f:
        json.dump(outputs, f, indent=4)
    # Average metrics across all entries   dd
    avg_precision = np.mean([o["precision_at_k"] for k, o in outputs.items()])
    avg_hit = np.mean([o["hit_at_1"] for k, o in outputs.items()])
    avg_recall = np.mean([o["recall_at_k"] for k, o in outputs.items()])

    print(f"Average AP: {avg_precision}")
    print(f"Average Hit@1: {avg_hit}")
    print(f"Average Recall@1: {avg_recall}")

    # Save averaged metrics to JSON
    avg_metrics = {
        "average_precision_at_k": avg_precision,
        "average_hit_at_1": avg_hit,
        "average_recall_at_k": avg_recall,
    }
    with open(os.path.join(result_folder, "average_metrics.json"), "w") as f:
        json.dump(avg_metrics, f, indent=4)

    print(
        f"Average metrics saved to {os.path.join(result_folder, 'average_metrics.json')}"
    )
