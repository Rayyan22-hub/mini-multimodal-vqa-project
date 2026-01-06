# Multimodal Story-Image Alignment

**Course:** Deep Neural Networks and Learning Systems  
**Project:** Story-Image Verification with Multimodal Learning

---

## Overview

This project tackles a pretty interesting problem in AI - figuring out if images actually match their story descriptions. Baseline models sometimes just accept whatever you show them, missing important objects or relationships. My goal here was to build something that checks more carefully using multiple verification methods.

The main challenge is that stories can be abstract while images are concrete, so you need different ways to verify alignment. I combined object detection, semantic similarity, and spatial reasoning to get better results.

## Dataset

Using the **StoryReasoning** dataset from HuggingFace:

- Dataset link: https://huggingface.co/datasets/daniel3303/StoryReasoning
- Contains stories with multiple image frames
- Training split used with 70/15/15 train/val/test distribution
- Fixed random seed (42) for reproducibility

## Methods

### 1. Object Detection (Scanner Step)

First step was detecting what's actually in the images using Faster R-CNN:

- Pre-trained **Faster R-CNN with ResNet-50 FPN backbone**
- Detects objects from 80 COCO categories
- Returns bounding boxes and confidence scores
- Maps detected objects to story keywords

The idea is simple - if the story mentions "person watching TV" then we should find both a person and a TV in the image.

### 2. Vision-Language Alignment (CLIP Step)

Object detection alone isn't enough because stories describe concepts, not just objects. That's where CLIP comes in:

- Uses **OpenAI's CLIP model** (ViT-B/32)
- Computes semantic similarity between images and text
- Better at understanding abstract concepts
- Frozen encoder (only train the fusion head to save time)

CLIP catches things like "relaxing" or "watching" that pure object detection misses.

### 3. Spatial Grounding Logic

One thing I noticed - some images have all the right objects but in wrong positions. Like if the story says "person sitting on couch watching TV" but the detected person is far from both the couch and TV, that's probably wrong.

My spatial scoring:

- Checks if related objects are physically near each other
- Uses bounding box center points and distances
- Common relationships: person+TV, person+furniture, person+toys
- Distance threshold set to 400 pixels empirically

### 4. Fusion MLP Classifier

The final piece combines all three scores into one prediction:

```python
class FusionMLP(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
```

**Input features:**

1. Object match score (0-1)
2. CLIP similarity score (0-1)
3. Spatial grounding score (0-1)

**Output:** Probability of story-image alignment (0-1)

### Training Details

- **Optimizer:** Adam (lr=0.001)
- **Loss function:** Binary Cross Entropy
- **Epochs:** 3 (more would probably help)
- **Batch processing:** Sequential to save GPU memory
- **Dataset split:** 70% train, 15% val, 15% test
- **Seed:** 42 for reproducibility

Kept it simple since this is mainly a proof of concept. The fusion head has around 5K parameters so it trains pretty fast even on Colab's free tier.

## Results

### Preliminary Metrics

Based on initial testing with ~120 samples:

| Metric           | Value |
| ---------------- | ----- |
| Train Accuracy   | ~72%  |
| Train F1 (macro) | ~0.68 |
| Val Accuracy     | ~63%  |
| Val F1 (macro)   | ~0.60 |

### Component Performance

Average scores on test set:

- **Object Detection Alignment:** ~0.65
- **CLIP Similarity:** ~0.58
- **Spatial Grounding:** ~0.52
- **Overall Fusion Score:** ~0.68

The object detection component worked best, which makes sense since it's the most direct check. Spatial grounding was trickier because the dataset has variable image sizes and frame compositions.

### Evaluation Approach

Implemented comprehensive metrics:

- Precision, Recall, F1-score
- Confusion matrix visualization
- Score distribution plots for each component
- Per-frame analysis with bounding box visualizations

Honestly the hardest part was getting all three components to work together smoothly without running out of GPU memory on Colab.

## Code Structure

The notebook is organized into chapters:

1. **Data Preparation** - Loading dataset, preprocessing, augmentation
2. **Model Architectures** - Encoder/decoder definitions, attention mechanisms
3. **Training Routines** - Loss functions, optimization, checkpointing
4. **Object Detection** - Faster R-CNN integration, visual tagging
5. **CLIP Integration** - Vision-language similarity
6. **Logic & Fusion** - Spatial grounding, MLP classifier
7. **Training** - Full dataset training with proper splits
8. **Evaluation** - Comprehensive metrics and visualizations
9. **Results** - Visual dashboards, sample predictions
10. **Summary** - Architecture overview and findings

Everything's in one notebook to make it easier to run on Colab.

## Challenges & Learnings

**What worked well:**

- Faster R-CNN detections were pretty accurate for common objects
- CLIP helped catch semantic mismatches that objects alone missed
- Three-way fusion gave better results than any single component

**What was difficult:**

- GPU memory management with multiple large models
- Spatial grounding rules needed lots of tuning
- CLIP truncation issues with long stories (fixed by limiting to 150 chars)
- Balancing the three scores in the fusion layer

**Key learning:** Multimodal problems really do need multiple verification methods. No single approach caught everything.

## Future Improvements

If I had more time/resources:

1. **Better spatial reasoning** - Current pixel distance approach is crude, could use relative positions or scene graphs
2. **More training** - Only did 3 epochs due to time, more would help
3. **Hyperparameter tuning** - Used reasonable defaults but didn't optimize systematically
4. **Ensemble methods** - Could try multiple fusion strategies
5. **Attention mechanisms** - Add attention over image regions that match story parts
6. **Data augmentation** - Could generate negative examples for better training

## Running the Code

1. Open `final v2.ipynb` in Google Colab
2. Mount Google Drive when prompted (for checkpoints)
3. Run cells in order - should take ~30-45 mins total
4. Models will download automatically (Faster R-CNN, CLIP)
5. Results saved to `/content/gdrive/MyDrive/DL_Checkpoints/`

**Requirements:**

- Google Colab
- ~8GB GPU memory
- Stable internet for model downloads

## References

- **Dataset:** StoryReasoning (daniel3303/StoryReasoning)
- **Faster R-CNN:** torchvision.models.detection.fasterrcnn_resnet50_fpn
- **CLIP:** OpenAI CLIP (openai/clip-vit-base-patch32)
- **Framework:** PyTorch 2.0+, transformers, datasets

---
