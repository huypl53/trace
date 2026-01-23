#!/bin/bash
# Line Segmentation Data Preparation Pipeline
# This script runs the full data preparation pipeline for line segmentation training.
#
# Pipeline steps:
# 1. Augment canvas JSON (layout/style/text mutations + negative examples)
# 2. Generate synthetic table images from augmented JSON
# 3. Prepare line dataset with masks from original images (optional)
# 4. Tile mask dataset into near-square patches
#
# Usage:
#   ./scripts/prepare_line_data_pipeline.sh [OPTIONS]
#
# Examples:
#   # Run full pipeline with defaults
#   ./scripts/prepare_line_data_pipeline.sh --input_dir data/raw_canvas
#
#   # Skip augmentation, use existing augmented data
#   ./scripts/prepare_line_data_pipeline.sh --input_dir data/raw_canvas --skip_augment
#
#   # Only run synthetic generation (no tiling)
#   ./scripts/prepare_line_data_pipeline.sh --input_dir data/raw_canvas --skip_tile

set -e  # Exit on error

# =============================================================================
# Default Configuration
# =============================================================================
INPUT_DIR=""
OUTPUT_BASE="data/line_seg_pipeline"
NUM_AUG=3
SEED=42
SPLIT="0.8 0.1 0.1"
TILE_SIZE=1280
TILE_STRIDE=1024
MAX_ASPECT_RATIO=1.2

# Augmentation probabilities
REMOVE_OUTER_BORDERS_PROB=0.3
REMOVE_INTERNAL_BORDERS_PROB=0.2
BG_CONTRAST_PROB=0.2
DENSE_TEXT_PROB=0.15
NARROW_COL_PROB=0.1

# Pipeline control
SKIP_AUGMENT=false
SKIP_SYNTHETIC=false
SKIP_PREPARE=false
SKIP_TILE=false
FILL_EMPTY_TEXT=true
RECURSIVE=false

# Smart tiling options
MIN_LINE_PIXELS=100
CONTENT_BBOX=true
BBOX_PADDING=50

# =============================================================================
# Parse Arguments
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output_base)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --num_aug)
            NUM_AUG="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2 $3 $4"
            shift 4
            ;;
        --tile_size)
            TILE_SIZE="$2"
            shift 2
            ;;
        --tile_stride)
            TILE_STRIDE="$2"
            shift 2
            ;;
        --skip_augment)
            SKIP_AUGMENT=true
            shift
            ;;
        --skip_synthetic)
            SKIP_SYNTHETIC=true
            shift
            ;;
        --skip_prepare)
            SKIP_PREPARE=true
            shift
            ;;
        --skip_tile)
            SKIP_TILE=true
            shift
            ;;
        --no_fill_empty_text)
            FILL_EMPTY_TEXT=false
            shift
            ;;
        --remove_outer_borders_prob)
            REMOVE_OUTER_BORDERS_PROB="$2"
            shift 2
            ;;
        --remove_internal_borders_prob)
            REMOVE_INTERNAL_BORDERS_PROB="$2"
            shift 2
            ;;
        --bg_contrast_prob)
            BG_CONTRAST_PROB="$2"
            shift 2
            ;;
        --dense_text_prob)
            DENSE_TEXT_PROB="$2"
            shift 2
            ;;
        --narrow_col_prob)
            NARROW_COL_PROB="$2"
            shift 2
            ;;
        --min_line_pixels)
            MIN_LINE_PIXELS="$2"
            shift 2
            ;;
        --no_content_bbox)
            CONTENT_BBOX=false
            shift
            ;;
        --bbox_padding)
            BBOX_PADDING="$2"
            shift 2
            ;;
        --recursive)
            RECURSIVE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 --input_dir <path> [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --input_dir <path>        Input directory with canvas JSON files"
            echo ""
            echo "Optional:"
            echo "  --output_base <path>      Base output directory (default: data/line_seg_pipeline)"
            echo "  --num_aug <n>             Number of augmentations per input (default: 3)"
            echo "  --seed <n>                Random seed (default: 42)"
            echo "  --split <train> <val> <test>  Split ratios (default: 0.8 0.1 0.1)"
            echo "  --tile_size <n>           Tile size in pixels (default: 1280)"
            echo "  --tile_stride <n>         Tile stride in pixels (default: 1024)"
            echo ""
            echo "Augmentation probabilities:"
            echo "  --remove_outer_borders_prob <p>    (default: 0.3)"
            echo "  --remove_internal_borders_prob <p> (default: 0.2)"
            echo "  --bg_contrast_prob <p>             (default: 0.2)"
            echo "  --dense_text_prob <p>              (default: 0.15)"
            echo "  --narrow_col_prob <p>              (default: 0.1)"
            echo ""
            echo "Pipeline control:"
            echo "  --skip_augment            Skip step 1 (augmentation)"
            echo "  --skip_synthetic          Skip step 2 (synthetic image generation)"
            echo "  --skip_prepare            Skip step 3 (prepare line dataset from originals)"
            echo "  --skip_tile               Skip step 4 (tiling)"
            echo "  --no_fill_empty_text      Don't fill empty cells with random text"
            echo "  --recursive               Process subdirectories recursively"
            echo ""
            echo "Smart tiling options:"
            echo "  --min_line_pixels <n>     Skip tiles with fewer line pixels (default: 100)"
            echo "  --no_content_bbox         Don't restrict tiling to content bounding box"
            echo "  --bbox_padding <n>        Padding around content bbox (default: 50)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$INPUT_DIR" ]]; then
    echo "Error: --input_dir is required"
    echo "Run with --help for usage information"
    exit 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# =============================================================================
# Setup Output Directories
# =============================================================================
AUGMENTED_DIR="${OUTPUT_BASE}/augmented_json"
SYNTHETIC_DIR="${OUTPUT_BASE}/synthetic_images"
PREPARED_DIR="${OUTPUT_BASE}/line_mask_dataset"
TILED_DIR="${OUTPUT_BASE}/line_mask_dataset_tiled"

echo "=============================================="
echo "Line Segmentation Data Preparation Pipeline"
echo "=============================================="
echo "Input directory:  $INPUT_DIR"
echo "Output base:      $OUTPUT_BASE"
echo "Augmentations:    $NUM_AUG per input"
echo "Split ratios:     $SPLIT"
echo "Tile size:        ${TILE_SIZE}x${TILE_SIZE} (stride: $TILE_STRIDE)"
echo "Recursive:        $RECURSIVE"
echo "=============================================="
echo ""

# =============================================================================
# Step 1: Augment Canvas JSON
# =============================================================================
if [[ "$SKIP_AUGMENT" == "false" ]]; then
    echo "[Step 1/4] Augmenting canvas JSON..."
    echo "  Output: $AUGMENTED_DIR"

    FILL_TEXT_FLAG=""
    if [[ "$FILL_EMPTY_TEXT" == "true" ]]; then
        FILL_TEXT_FLAG="--fill_empty_text"
    fi

    RECURSIVE_FLAG=""
    if [[ "$RECURSIVE" == "true" ]]; then
        RECURSIVE_FLAG="--recursive"
    fi

    uv run python -m scripts.augment_canvas_data \
        --input_dir "$INPUT_DIR" \
        --output_dir "$AUGMENTED_DIR" \
        --num_aug "$NUM_AUG" \
        $FILL_TEXT_FLAG \
        $RECURSIVE_FLAG \
        --remove_outer_borders_prob "$REMOVE_OUTER_BORDERS_PROB" \
        --remove_internal_borders_prob "$REMOVE_INTERNAL_BORDERS_PROB" \
        --bg_contrast_prob "$BG_CONTRAST_PROB" \
        --dense_text_prob "$DENSE_TEXT_PROB" \
        --narrow_col_prob "$NARROW_COL_PROB"

    echo "  Done!"
    echo ""
else
    echo "[Step 1/4] Skipping augmentation (--skip_augment)"
    AUGMENTED_DIR="$INPUT_DIR"
    echo ""
fi

# =============================================================================
# Step 2: Generate Synthetic Table Images from Augmented JSON
# =============================================================================
if [[ "$SKIP_SYNTHETIC" == "false" ]]; then
    echo "[Step 2/4] Generating synthetic table images..."
    echo "  Input:  $AUGMENTED_DIR"
    echo "  Output: $SYNTHETIC_DIR"

    # Note: Always use --recursive for synthetic since augmented output is flat
    # but may contain files from nested input dirs
    uv run python -m scripts.draw_table_from_json \
        --input_dir "$AUGMENTED_DIR" \
        --output_dir "$SYNTHETIC_DIR" \
        --recursive \
        --split $SPLIT \
        --seed "$SEED"

    echo "  Done!"
    echo ""
else
    echo "[Step 2/4] Skipping synthetic image generation (--skip_synthetic)"
    echo ""
fi

# =============================================================================
# Step 3: Prepare Line Dataset with Masks (from original images)
# =============================================================================
if [[ "$SKIP_PREPARE" == "false" ]]; then
    echo "[Step 3/4] Preparing line dataset with masks..."
    echo "  Input:  $INPUT_DIR (original canvas + images)"
    echo "  Output: $PREPARED_DIR"

    uv run python -m scripts.prepare_line_dataset \
        --input_dir "$INPUT_DIR" \
        --output_dir "$PREPARED_DIR" \
        --padding 5 \
        --output_mode mask \
        --split $SPLIT \
        --seed "$SEED"

    echo "  Done!"
    echo ""
else
    echo "[Step 3/4] Skipping line dataset preparation (--skip_prepare)"
    echo ""
fi

# =============================================================================
# Step 4: Tile Mask Dataset into Near-Square Patches
# =============================================================================
if [[ "$SKIP_TILE" == "false" ]]; then
    # Determine which dataset to tile (prefer synthetic if available)
    if [[ "$SKIP_SYNTHETIC" == "false" && -d "$SYNTHETIC_DIR" ]]; then
        TILE_INPUT="$SYNTHETIC_DIR"
        TILE_OUTPUT="${SYNTHETIC_DIR}_tiled"
    elif [[ "$SKIP_PREPARE" == "false" && -d "$PREPARED_DIR" ]]; then
        TILE_INPUT="$PREPARED_DIR"
        TILE_OUTPUT="$TILED_DIR"
    else
        echo "[Step 4/4] Skipping tiling (no dataset to tile)"
        SKIP_TILE=true
    fi

    if [[ "$SKIP_TILE" == "false" ]]; then
        echo "[Step 4/4] Tiling mask dataset into patches (smart tiling)..."
        echo "  Input:  $TILE_INPUT"
        echo "  Output: $TILE_OUTPUT"
        echo "  Tile size: ${TILE_SIZE}, stride: ${TILE_STRIDE}"
        echo "  Min line pixels: ${MIN_LINE_PIXELS}, content bbox: ${CONTENT_BBOX}"

        CONTENT_BBOX_FLAG=""
        if [[ "$CONTENT_BBOX" == "true" ]]; then
            CONTENT_BBOX_FLAG="--content_bbox"
        fi

        uv run python -m scripts.tile_line_mask_dataset \
            --input_dir "$TILE_INPUT" \
            --output_dir "$TILE_OUTPUT" \
            --tile_size "$TILE_SIZE" \
            --tile_stride "$TILE_STRIDE" \
            --max_aspect_ratio "$MAX_ASPECT_RATIO" \
            --min_line_pixels "$MIN_LINE_PIXELS" \
            --bbox_padding "$BBOX_PADDING" \
            --content_bbox \
            $CONTENT_BBOX_FLAG

        echo "  Done!"
        echo ""
    fi
else
    echo "[Step 4/4] Skipping tiling (--skip_tile)"
    echo ""
fi

# =============================================================================
# Summary
# =============================================================================
echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo "Output directories:"
if [[ "$SKIP_AUGMENT" == "false" ]]; then
    echo "  Augmented JSON:    $AUGMENTED_DIR"
fi
if [[ "$SKIP_SYNTHETIC" == "false" ]]; then
    echo "  Synthetic images:  $SYNTHETIC_DIR"
fi
if [[ "$SKIP_PREPARE" == "false" ]]; then
    echo "  Line mask dataset: $PREPARED_DIR"
fi
if [[ "$SKIP_TILE" == "false" && -n "$TILE_OUTPUT" ]]; then
    echo "  Tiled dataset:     $TILE_OUTPUT"
fi
echo ""
echo "Next step: Train with:"
echo "  uv run python train.py --task line --line_data_mode mask \\"
echo "      --config_file configs/train_line_mask.json"
echo "=============================================="
