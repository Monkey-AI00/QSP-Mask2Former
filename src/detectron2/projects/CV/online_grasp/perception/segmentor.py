"""Segmentor wrapper for behavior-compatible refactor phase."""

from __future__ import annotations

import os

import mecheye_live_pointrend_pointcloud_shape_prior as live_utils


def _load_predictor(args):
    score_thr = float(args.score_thr)
    model_family = str(args.model_family).strip().lower()
    prior_path_override = str(args.shape_prior_npy).strip()
    if prior_path_override:
        prior_path_override = live_utils._require_existing_file(prior_path_override, "prior npy")

    if model_family == "mask2former":
        config_prior = live_utils._require_existing_file(
            str(args.config_file_prior).strip() or live_utils._DEFAULT_MASK2FORMER_QSP_CONFIG,
            "Mask2Former prior config",
        )
        weights_prior = live_utils._require_existing_file(str(args.weights_prior).strip(), "prior weights")
        predictor = live_utils.build_mask2former_predictor(
            mask2former_root=str(args.mask2former_root),
            config_file=config_prior,
            weights=weights_prior,
            score_thresh=score_thr,
            device=str(args.device),
            num_classes=int(args.num_classes),
            prior_path_override=prior_path_override,
        )
    else:
        config = live_utils._require_existing_file(
            str(args.config_file).strip() or live_utils._DEFAULT_POINTREND_CONFIG,
            "PointRend config",
        )
        weights_prior = live_utils._require_existing_file(str(args.weights_prior).strip(), "prior weights")
        if prior_path_override:
            os.environ["SHAPE_PRIOR_PATH"] = str(prior_path_override)
        predictor = live_utils.build_pointrend_predictor(
            config_file=config,
            weights=weights_prior,
            mask_head_name="ShapeAwareCoarseMaskHead",
            score_thresh=score_thr,
            device=str(args.device),
            num_classes=int(args.num_classes),
        )
    return predictor


class Segmentor:
    """Thin wrapper around legacy predictor construction."""

    def __init__(self, args, predictor=None):
        self.args = args
        self.predictor = predictor if predictor is not None else _load_predictor(args)

    def infer_mask_from_frame(self, color_bgr):
        out = self.predictor(color_bgr)
        inst = out["instances"].to("cpu")
        _, mask_pc, invert_applied = live_utils._build_output_masks(
            inst,
            mask_mode=str(self.args.mask_mode),
            pc_mask_mode=str(self.args.pc_mask_mode),
            pc_iou_thresh=float(self.args.pc_iou_thresh),
            pc_join_dilate=int(self.args.pc_join_dilate),
            mask_close=int(self.args.mask_close),
            mask_dilate=int(self.args.mask_dilate),
            mask_erode=int(self.args.mask_erode),
            invert_mask=bool(self.args.invert_mask),
            auto_invert_mask=bool(self.args.auto_invert_mask),
        )
        print("[mask][pc ] " + live_utils._mask_stats(mask_pc) + f" invert={invert_applied}")
        return mask_pc


__all__ = ["Segmentor", "_load_predictor"]

