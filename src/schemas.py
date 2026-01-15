from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import uuid

class SelectorInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    image: Any  # PIL Image
    text: Optional[str] = None
    class_name_override: Optional[str] = None
    input_boxes: List[List[int]] = []  # [[x1, y1, x2, y2], ...]
    input_labels: List[int] = []       # [1, 0, ...] 1=Include, 0=Exclude
    crop_box: Optional[List[int]] = None # [x1, y1, x2, y2]
    
class ObjectState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    object_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    score: float

    # To make mutable in the future
    class_name: str
    anchor_box: List[int] # [x1, y1, x2, y2] - STATIC from Selector
 
    # Numpy array - DYNAMIC (Selector -> Refiner)
    binary_mask: Any
    
    # Backup for Undo (Selector result)
    initial_mask: Any = None
    
    # Refinement History
    input_points: List[List[int]] = []
    input_labels: List[int] = []

class GlobalStore(BaseModel):
    image_path: Optional[str] = None
    objects: Dict[str, ObjectState] = {}

class ProjectState(BaseModel):
    playlist: List[str] = []
    current_index: int = -1
    annotations: Dict[str, GlobalStore] = {}
    prompt_history: List[str] = []
    class_name_history: List[str] = []
