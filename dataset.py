import enum
import random
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torchvision import transforms


class TargetType(enum.Enum):
    """
    Enum for the target type of the dataset.
    """
    ORIGINAL = enum.auto()
    DIFFERENCE = enum.auto()


class FrameMode(enum.Enum):
    """
    Clear distinction between different frame processing modes.
    """
    ALTERNATING_WITH_GT = "alternating_with_gt"  # Alternating frames with GT available (formerly full_fps_conditioning=False)
    CONSECUTIVE_NO_GT = "consecutive_no_gt"      # Consecutive frames with no GT (formerly full_fps_conditioning=True)


class FrameProcessingResult(NamedTuple):
    """
    result of frame processing
    """
    conditioning_frames: torch.Tensor  # Conditioning frames for model input
    target_frames: torch.Tensor       # Ground truth (when available) or dummy
    has_ground_truth: bool            # Whether GT is available
    chunk_size_required: int          # Minimum frames required for this mode


class FrameProcessor:
    """
    frame processing logic
    """
    
    def __init__(self, mode: FrameMode, num_target_frames: int):
        self.mode = mode
        self.num_target_frames = num_target_frames
    
    def get_required_chunk_size(self) -> int:
        """Minimum frames required per mode"""
        if self.mode == FrameMode.ALTERNATING_WITH_GT:
            return self.num_target_frames * 2 + 1  # Alternating frames + ensure GT
        else:  # CONSECUTIVE_NO_GT
            return self.num_target_frames + 1      # Consecutive frames only
    
    def process_frames(self, frames) -> FrameProcessingResult:
        """
        Process frames according to mode
        
        Args:
            frames: Original frames array/tensor (F, C, H, W)
            
        Returns:
            FrameProcessingResult with clearly separated conditioning and target frames
        """
        if self.mode == FrameMode.ALTERNATING_WITH_GT:
            return self._process_alternating_with_gt(frames)
        else:  # CONSECUTIVE_NO_GT
            return self._process_consecutive_no_gt(frames)
    
    def _process_alternating_with_gt(self, frames) -> FrameProcessingResult:
        """
        Alternating-frame mode: even indices are conditioning, odd indices are GT
        
        Example: [0,1,2,3,4,5,6] -> conditioning=[0,2,4,6], target=[1,3,5]
        """
        conditioning_frames = torch.tensor(frames[::2])  # even indices (0,2,4,6,...)
        target_frames = torch.tensor(frames[1::2])       # odd indices (1,3,5,...)
        
        return FrameProcessingResult(
            conditioning_frames=conditioning_frames,
            target_frames=target_frames,
            has_ground_truth=True,
            chunk_size_required=self.get_required_chunk_size()
        )
    
    def _process_consecutive_no_gt(self, frames) -> FrameProcessingResult:
        """
        Consecutive-frame mode: use all frames as conditioning, no GT
        
        Example: [0,1,2,3,4,5,6] -> conditioning=[0,1,2,3,4,5,6], target=dummy
        """
        conditioning_frames = torch.tensor(frames)  # all frames
        # Dummy target (unused by the model)
        dummy_target = torch.zeros(
            (self.num_target_frames,) + frames.shape[1:], 
            dtype=torch.uint8
        )
        
        return FrameProcessingResult(
            conditioning_frames=conditioning_frames,
            target_frames=dummy_target,
            has_ground_truth=False,
            chunk_size_required=self.get_required_chunk_size()
        )


def _is_tensor_video_clip(clip):
    if not torch.is_tensor(clip):
        raise TypeError("clip should be Tensor. Got %s" % type(clip))

    if not clip.ndimension() == 4:
        raise ValueError("clip should be 4D. Got %dD" % clip.dim())

    return True


def to_tensor(clip):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (T, C, H, W)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
    """
    _is_tensor_video_clip(clip)
    return clip.float() / 255.0


class ToTensorVideo:
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    """

    def __init__(self):
        pass

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor, dtype=torch.uint8): Size is (T, C, H, W)
        Return:
            clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
        """
        return to_tensor(clip)

    def __repr__(self) -> str:
        return self.__class__.__name__


def hflip(clip):
    """
    Args:
        clip (torch.tensor): Video clip to be normalized. Size is (T, C, H, W)
    Returns:
        flipped clip (torch.tensor): Size is (T, C, H, W)
    """
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    return clip.flip(-1)


class ConsistentRandomHorizontalFlip:
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample["frames"] = hflip(sample["frames"])
            sample["cond_frames"] = hflip(sample["cond_frames"])
        return sample


def create_transform(image_size=512, horizontal_flip_prob=0.2):
    """
    Returns a function that applies a consistent random horizontal flip (with probability `horizontal_flip_prob`)
    and then transforms the individual images (resize, center crop, and convert to tensor).
    """
    image_transform = transforms.Compose(
        [
            ToTensorVideo(),
            transforms.Resize(image_size),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
            ),
        ]
    )

    def transform_sample(sample):
        sample = ConsistentRandomHorizontalFlip(p=horizontal_flip_prob)(sample)
        sample["frames"] = image_transform(sample["frames"])
        sample["cond_frames"] = image_transform(sample["cond_frames"])
        return sample

    return transform_sample


def interpolate_ecg_signal(cycle_signal, fixed_cycle_length):
    """
    Interpolates the cardiac cycle signal to a fixed length.
    """
    cycle_signal = np.array(cycle_signal, dtype=np.float32)
    n = len(cycle_signal)
    if n < 2:
        return np.full(
            fixed_cycle_length, cycle_signal[0] if n > 0 else 0, dtype=np.float32
        )
    old_indices = np.linspace(0, n - 1, num=n)
    new_indices = np.linspace(0, n - 1, num=fixed_cycle_length)
    interpolated_cycle = np.interp(new_indices, old_indices, cycle_signal).astype(
        np.float32
    )
    return interpolated_cycle


def normalize_ecg_signal(ecg_signal):
    min_val = ecg_signal.min()
    max_val = ecg_signal.max()
    if max_val - min_val == 0:
        return ecg_signal - min_val
    return (ecg_signal - min_val) / (max_val - min_val)


class AngioInterpolationDataset(torch.utils.data.Dataset):
    """
    Refactored Angio video interpolation dataset - clear frame processing logic
    """
    def __init__(
            self, 
            data_root_dir: Path | str, 
            image_size=512, 
            num_target_frames=15, 
            use_ecg=False, 
            ecg_signal_len=100, 
            test_mode=False,
            frame_mode="alternating_with_gt"  # Explicit parameter ('alternating_with_gt' or 'consecutive_no_gt')
        ):
        if isinstance(data_root_dir, str):
            data_root_dir = Path(data_root_dir)
        self.data_root_dir = data_root_dir
        self.video_paths = list(data_root_dir.glob("**/*.npz"))

        horizontal_flip_prob = 0.0 if test_mode else 0.2
        self.transform = create_transform(
            image_size=image_size, horizontal_flip_prob=horizontal_flip_prob
        )

        self.num_target_frames = num_target_frames 
        self.test_mode = test_mode
        self.use_ecg = use_ecg
        self.ecg_signal_len = ecg_signal_len
        
        if frame_mode == "alternating_with_gt":
            self.frame_mode = FrameMode.ALTERNATING_WITH_GT
        elif frame_mode == "consecutive_no_gt":
            self.frame_mode = FrameMode.CONSECUTIVE_NO_GT
        else:
            raise ValueError(f"Unknown frame_mode: {frame_mode}. Use 'alternating_with_gt' or 'consecutive_no_gt'")
        
        # Initialize frame processor
        self.frame_processor = FrameProcessor(self.frame_mode, num_target_frames)
        self.chunk_size = self.frame_processor.get_required_chunk_size()

    def get_chunk_size(self):
        """
        Compute chunk size based on frame mode (kept for backward compatibility)
        
        Returns:
            int: Minimum number of frames required to create at least one chunk
        """
        return self.frame_processor.get_required_chunk_size()
    
    def can_process_video(self, num_frames: int) -> bool:
        """
        Check if the video can be processed with the current mode
        """
        return num_frames >= self.chunk_size
    
    def get_frame_mode_info(self) -> dict:
        """
        Return current frame-mode information (for debugging)
        """
        return {
            'mode': self.frame_mode.value,
            'has_ground_truth': self.frame_mode == FrameMode.ALTERNATING_WITH_GT,
            'chunk_size': self.chunk_size,
            'description': {
                FrameMode.ALTERNATING_WITH_GT: "Alternating-frame mode - GT comparable",
                FrameMode.CONSECUTIVE_NO_GT: "Consecutive-frame mode - GT not comparable"
            }[self.frame_mode]
        }

    def random_crop(self, num_frames):
        current_chunk_size = self.chunk_size
        rand_end = max(0, num_frames - current_chunk_size)
        begin_index = random.randint(0, rand_end)  # [0, rand_end]
        end_index = begin_index + current_chunk_size - 1  # inclusive end index
        assert end_index - begin_index + 1 >= current_chunk_size
        return begin_index, end_index

    def _get_data(self, vframes, frame_start_idx, frame_end_idx, ecg_signal=None, ecg_samples_per_frame=None):
        """
        Process frame data to create a sample (refactored version)
        """
        sampled_frames = vframes[frame_start_idx: frame_end_idx + 1]  # +1 for inclusive end
        
        # New clearer frame processing
        result = self.frame_processor.process_frames(sampled_frames)
        
        data = {
            "frames": result.target_frames,  # GT or dummy
            "cond_frames": result.conditioning_frames,  # For model input
            "has_ground_truth": result.has_ground_truth,  # Whether GT exists (newly added)
            "processing_mode": self.frame_mode.value  # Processing mode (newly added)
        }
        
        # ECG processing
        if self.use_ecg:
            ecg_start_idx = int(frame_start_idx * ecg_samples_per_frame)
            ecg_end_idx = int((frame_end_idx + 1) * ecg_samples_per_frame)  # +1 for inclusive end
            cond_ecg = ecg_signal[ecg_start_idx:ecg_end_idx]
            cond_ecg = interpolate_ecg_signal(cond_ecg, self.ecg_signal_len)
            data["cond_ecg"] = torch.tensor(cond_ecg)
            
        return self.transform(data)

    def _prepare_training_sample(
        self, vframes, ecg_signal=None, ecg_samples_per_frame=None
    ):
        """Handle training mode sample preparation"""
        frame_start_idx, frame_end_idx = self.random_crop(len(vframes))
        sample = self._get_data(vframes, frame_start_idx, frame_end_idx, ecg_signal, ecg_samples_per_frame)
        return sample

    def _prepare_test_sample(
        self, vframes, ecg_signal=None, ecg_samples_per_frame=None
    ):
        """
        Handle test mode sample preparation with seamless chunk connection.
        Each chunk connects seamlessly with the next, and the last chunk is handled specially.
        """
        current_chunk_size = self.chunk_size

        if self.frame_mode == FrameMode.ALTERNATING_WITH_GT and len(vframes) % 2 == 0:
            # In alternating-frame mode, the total number of frames must be odd
            vframes = vframes[:-1]
        
        # Return empty chunks for videos that are too short
        if len(vframes) < current_chunk_size:
            return []  # Empty chunks list indicates video is too short
            
        total_frames = len(vframes)
        chunks = []
        
        # Calculate how many full chunks we can make
        current_start = 0
        chunk_idx = 0
        
        while current_start < total_frames:
            # Calculate end index for current chunk
            tentative_end = current_start + current_chunk_size - 1
            
            if tentative_end >= total_frames - 1:
                # Last chunk: use all remaining frames
                frame_start_idx = max(0, total_frames - current_chunk_size)
                frame_end_idx = total_frames - 1
                is_last_chunk = True
            else:
                # Regular chunk
                frame_start_idx = current_start
                frame_end_idx = tentative_end
                is_last_chunk = False
            
            # Calculate overlap with previous chunk
            if chunk_idx == 0:
                overlap_frames_in_merged = 0
            elif not is_last_chunk:
                # Regular chunks: always overlap by exactly 1 conditioning frame
                # In merged video, this becomes 1 frame to remove
                overlap_frames_in_merged = 1
            else:
                # Last chunk: simple calculation due to odd video length and odd chunk size
                prev_chunk = chunks[-1]
                prev_end = prev_chunk['frame_end_idx']
                overlap_frames_in_merged = max(0, prev_end - frame_start_idx + 1)
            
            # Create the chunk
            sample = self._get_data(vframes, frame_start_idx, frame_end_idx, ecg_signal, ecg_samples_per_frame)
            sample["chunk_idx"] = chunk_idx
            sample["frame_start_idx"] = frame_start_idx
            sample["frame_end_idx"] = frame_end_idx
            sample["overlap_frames"] = overlap_frames_in_merged
            sample["is_last_chunk"] = is_last_chunk
            chunks.append(sample)
            
            # If this was the last chunk, break immediately
            if is_last_chunk:
                break
            
            # Move to next chunk: start at the last conditioning frame of current chunk
            # This ensures exactly 1 conditioning frame overlap for regular chunks
            current_start = frame_end_idx
            chunk_idx += 1
        
        # Add total_chunks info to all chunks
        for chunk in chunks:
            chunk["total_chunks"] = len(chunks)
        
        return chunks

    def _load_video_data(self, path, load_ecg=False):
        data = np.load(path)
        vframes = data["pixel_array"]  # (F, H, W)
        vframes = np.repeat(np.expand_dims(vframes, axis=1), 3, axis=1)  # (F, 3, H, W)
        if load_ecg:
            ecg_signal = data["ecg_signal"]
            ecg_signal = normalize_ecg_signal(ecg_signal)
            ecg_samples_per_frame = len(ecg_signal) / len(vframes)
        else:
            ecg_signal = None
            ecg_samples_per_frame = None
        return vframes, ecg_signal, ecg_samples_per_frame

    def __getitem__(self, index):
        video_path = self.video_paths[index]
        vframes, ecg_signal, ecg_samples_per_frame = self._load_video_data(video_path, self.use_ecg)
        video_rel_path = video_path.relative_to(self.data_root_dir)
        key = ".".join(str(video_rel_path).split(".")[:-1]).replace("/", "_")

        if self.test_mode:
            chunks = self._prepare_test_sample(
                vframes, ecg_signal, ecg_samples_per_frame
            )
            return {
                "key": key,
                "chunks": chunks,
            }
        else:
            sample = self._prepare_training_sample(
                vframes, ecg_signal, ecg_samples_per_frame
            )
            sample["key"] = key
            return sample

    def __len__(self):
        return len(self.video_paths)


def create_alternating_dataset(data_path, num_frames=7, **kwargs):
    """
    Create dataset in alternating-frame mode (GT comparable)
    
    Args:
        data_path: Data path
        num_frames: Number of frames to generate
        **kwargs: Other AngioInterpolationDataset parameters
    
    Returns:
        AngioInterpolationDataset: Dataset configured for alternating-frame mode
    """
    return AngioInterpolationDataset(
        data_path, 
        num_target_frames=num_frames,
        frame_mode="alternating_with_gt",
        **kwargs
    )


def create_consecutive_dataset(data_path, num_frames=15, **kwargs):
    """
    Create dataset in consecutive-frame mode (GT not comparable)
    
    Args:
        data_path: Data path
        num_frames: Number of frames to generate
        **kwargs: Other AngioInterpolationDataset parameters
    
    Returns:
        AngioInterpolationDataset: Dataset configured for consecutive-frame mode
    """
    return AngioInterpolationDataset(
        data_path,
        num_target_frames=num_frames, 
        frame_mode="consecutive_no_gt",
        **kwargs
    )
