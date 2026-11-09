"""nuScenes val-split helpers.

The official 150-scene val split is shipped in nuscenes-devkit at
`nuscenes.utils.splits.val`. This module wraps it and exposes a token list.
"""
from typing import Dict, List, NamedTuple


class SampleMeta(NamedTuple):
    sample_token: str
    scene_name: str
    keyframe_idx: int  # 0-based index within the scene


def get_val_scene_names() -> List[str]:
    """Return the 150 official val scene names from nuscenes-devkit."""
    from nuscenes.utils.splits import val as val_scene_names
    return list(val_scene_names)


def get_val_sample_metadata(nusc) -> List[SampleMeta]:
    """All keyframes of the 150 official val scenes, scene-major then in sample
    order, with the scene name and the 0-based keyframe index."""
    val_scenes = set(get_val_scene_names())
    out: List[SampleMeta] = []
    for scene in nusc.scene:
        scene_name = scene["name"]
        if scene_name not in val_scenes:
            continue
        token = scene["first_sample_token"]
        idx = 0
        while token != "":
            out.append(SampleMeta(token, scene_name, idx))
            sample = nusc.get("sample", token)
            token = sample["next"]
            idx += 1
    return out


def get_val_metadata_by_token(nusc) -> Dict[str, SampleMeta]:
    """Token -> SampleMeta lookup for joining onto eval results."""
    return {m.sample_token: m for m in get_val_sample_metadata(nusc)}
