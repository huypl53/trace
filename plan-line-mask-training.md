# Plan: Mask-Based Line Training Loader

## Goal
- Add a training path that uses RGB images + binary mask images for line segmentation.
- Keep the existing JSON-based line dataset working as-is.

## Open Questions (Need Your Confirm)
- Mask format: single-channel (all lines) or two-channel (horizontal/vertical split)?
- If two-channel, is it one 2/3-channel image or two files (e.g., `*_mask_h.png` + `*_mask_v.png`)?
- Dataset layout: same folder with naming convention, or separate `images/` and `masks/` dirs?
- Should `output_ch` be 1 or 2 for this training, and do you want inference updated if 1?

## Proposed Steps
1. Define pairing rules for image ↔ mask (folder structure + naming), and add a small validator to catch missing pairs.
2. Implement a new dataset/loader class (e.g., `LineMask_Dataset`) that:
   - Loads RGB image and mask(s).
   - Resizes image with bilinear and mask(s) with nearest-neighbor.
   - Normalizes mask(s) to [0, 1] with a safe max check (avoid div-by-zero).
   - Produces `gt_weight` as all-ones like the current line loader.
3. Add a dedicated transform (or extend `LineAugmentation`) to apply the same resize to image + mask(s).
4. Wire `train.py` to select the new loader via a CLI/config flag (keep default as JSON).
5. Update docs (likely `LINE_SEGMENTATION_CANVAS.md` or a new short doc) with expected directory layout + examples.
6. Optional: update `line_infer.py`/evaluation if you want to support `output_ch=1` (single mask).

## Validation
- Run a quick sample load: print shapes + min/max of masks to confirm normalization.
- (Optional) Save a debug grid for one batch to verify alignment.
