import numpy as np
from concurrent.futures import ThreadPoolExecutor


def _get_region_patches(state, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    ps = state.ps

    px1, py1 = x1 // ps, y1 // ps
    px2 = min((x2 - 1) // ps + 1, state.cols)
    py2 = min((y2 - 1) // ps + 1, state.rows)

    indices = []
    for r in range(py1, py2):
        for c in range(px1, px2):
            indices.append(r * state.cols + c)
    return indices


def classify_objects(objects, target_states, class_states, max_workers=4):
    """
    objects: list of (img_idx, obj_idx, bbox, det_conf, prompt)
    target_states: list[ImageEmbeddingState]
    class_states: dict[str, ImageEmbeddingState]
    returns: list of (img_idx, obj_idx, label, score, bbox, det_conf, prompt)
    """

    def classify_one(obj):
        img_idx, obj_idx, bbox, det_conf, prompt = obj
        target = target_states[img_idx]
        region_idx = _get_region_patches(target, bbox)

        if not region_idx:
            return None

        region_embs = target.Xn[region_idx]  # (N_region, D)
        best_label = None
        best_score = -1.0

        for cname, cstate in class_states.items():
            sim = region_embs @ cstate.Xn.T  # (N_region, N_class_patches)
            score = float(np.max(sim))
            if score > best_score:
                best_score = score
                best_label = cname

        return img_idx, obj_idx, best_label, best_score, bbox, det_conf, prompt

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        res = list(ex.map(classify_one, objects))
    return [r for r in res if r is not None]
