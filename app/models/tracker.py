from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from app.core.config import DEVICE, SAM2_CHECKPOINT, SAM2_CONFIG


class SAM2Tracker:
    def __init__(
        self,
        config: str = None,
        checkpoint_path: str = None,
        device: str = None,
    ):
        self.config = config or SAM2_CONFIG
        self.checkpoint_path = checkpoint_path or str(SAM2_CHECKPOINT)
        self.device = device or DEVICE
        self.state = None
        self.initialized = False

        sam2_model = build_sam2_video_predictor(
            self.config,
            self.checkpoint_path,
            device=self.device,
        )
        
        self.tracker = SAM2VideoPredictor(sam2_model)


    def init(self, frame_dir: str):
        self.state = self.tracker.init_state(video_path=frame_dir)
        self.initialized = True

    def track():
        pass

    def add_object_with_box(self, frame_idx, obj_id, box):
        self.predictor.add_new_points_or_box(
            inference_state=self.state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            box=box,
        )

    def propagate(self):
        return self.predictor.propagate_in_video(self.state)
