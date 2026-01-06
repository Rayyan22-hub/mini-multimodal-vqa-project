# Multimodal Story Image Alignment

**Course:** Deep Neural Networks and Learning Systems  
**Project:** Story Image Verification with Multimodal Learning

---

## Overview

This project addresses the problem of determining whether images truly match their corresponding story descriptions. Baseline vision language models often accept images even when important objects or relationships described in the story are missing or incorrect. The goal of this work was to design a more careful verification system that evaluates alignment using multiple complementary signals.

A central challenge is that stories are often abstract, while images are concrete. Because of this gap, no single method is sufficient. To address the problem, the project combines object detection, semantic similarity, and spatial reasoning to produce more reliable story image alignment decisions.

---

## Dataset

The project uses the **StoryReasoning** dataset from HuggingFace.

- Dataset link: https://huggingface.co/datasets/daniel3303/StoryReasoning  
- Contains stories paired with multiple image frames  
- Dataset split: 70 percent training, 15 percent validation, 15 percent test  
- Fixed random seed of 42 for reproducibility  

---

## Methods

### 1. Object Detection (Scanner Step)

The first step identifies which objects are present in each image using Faster R CNN.

- Pretrained Faster R CNN with a ResNet 50 FPN backbone  
- Detects objects from 80 COCO categories  
- Outputs bounding boxes and confidence scores  
- Detected objects are mapped to keywords extracted from the story  

For example, if the story mentions a person watching television, the image should reasonably contain both a person and a television.

---

### 2. Vision Language Alignment (CLIP Step)

Object detection alone cannot capture abstract concepts or actions described in stories. To address this limitation, CLIP is used to measure semantic similarity.

- Uses OpenAI CLIP ViT B 32  
- Computes similarity between image and story text embeddings  
- Effective at capturing abstract concepts such as actions or context  
- CLIP encoders are frozen and only the fusion head is trained  

This component helps detect mismatches such as an image containing the right objects but depicting an unrelated situation.

---

### 3. Spatial Grounding Logic

Some images contain the correct objects but arrange them in implausible ways. For instance, a story may describe a person sitting on a couch watching television, while the detected person is far from both the couch and the television.

The spatial grounding score:

- Computes distances between bounding box center points  
- Focuses on common relationships such as person and television or person and furniture  
- Uses a fixed pixel distance threshold of 400 selected empirically  

This step helps reject visually inconsistent alignments.

---

### 4. Fusion MLP Classifier

The final decision is produced by combining all three signals using a lightweight multilayer perceptron.

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

1. Object match score between 0 and 1  
2. CLIP similarity score between 0 and 1  
3. Spatial grounding score between 0 and 1  

**Output:** Probability that the image aligns with the story

---

## Training Details

- Optimizer: Adam with learning rate 0.001  
- Loss function: Binary Cross Entropy  
- Epochs: 10  
- Batch processing: Sequential to reduce GPU memory usage  
- Dataset split: 70 percent training, 15 percent validation, 15 percent test  
- Random seed: 42  

Training for 10 epochs allowed the fusion classifier to converge more stably compared to earlier experiments with fewer epochs. Validation loss stabilized after approximately 7 to 8 epochs, with no strong signs of overfitting observed due to the small size of the fusion network and frozen encoders.

The fusion head contains roughly 5K parameters, so training remains efficient and feasible on Colab’s free GPU tier.

---

## Results

### Preliminary Metrics

After training for 10 epochs and evaluating on approximately 120 samples:

| Metric           | Value |
|------------------|-------|
| Train Accuracy   | ~78%  |
| Train F1 (macro) | ~0.74 |
| Val Accuracy     | ~69%  |
| Val F1 (macro)   | ~0.66 |

Compared to earlier experiments with fewer epochs, both accuracy and F1 score improved, particularly on the validation set, indicating better calibration of the fusion model.

---

### Component Performance

Average scores on the test set:

- Object Detection Alignment: ~0.68  
- CLIP Similarity: ~0.61  
- Spatial Grounding: ~0.56  
- Overall Fusion Score: ~0.72  

Object detection remains the strongest individual signal. CLIP benefits indirectly from longer training through improved weighting in the fusion layer. Spatial grounding improves slightly but remains the most challenging component due to variation in image scale and framing.

---

### Evaluation Approach

Evaluation included:

- Precision, recall, and F1 score  
- Confusion matrix visualization  
- Score distribution plots for each component  
- Per frame qualitative analysis with bounding box visualizations  

Training curves indicate diminishing performance gains beyond approximately 8 epochs, suggesting that longer training primarily improves fusion stability rather than raw component performance.

---

## Code Structure

The notebook is organized into the following sections:

1. Data preparation and preprocessing  
2. Model architectures  
3. Training routines and optimization  
4. Object detection integration  
5. CLIP based vision language similarity  
6. Spatial grounding and fusion logic  
7. Full training with fixed dataset splits  
8. Evaluation and metrics  
9. Result visualization and sample predictions  
10. Summary and conclusions  

All components are contained in a single notebook for ease of execution in Google Colab.

---

## Challenges and Learnings

**What worked well:**

- Faster R CNN provided reliable detections for common objects  
- CLIP captured semantic mismatches missed by object detection  
- Longer training improved fusion stability and calibration  

**What was difficult:**

- GPU memory constraints with multiple large models  
- Sensitivity of spatial grounding thresholds  
- Handling CLIP input length limits for long stories  
- Limited gains beyond 8 epochs due to frozen encoders  

**Key takeaway:** Multimodal verification benefits significantly from combining multiple reasoning strategies, while gains from longer training are primarily realized in the fusion stage.

---

## Future Improvements

With epoch count explored, future work would focus on architectural improvements rather than longer training:

1. Learnable spatial reasoning instead of fixed pixel thresholds  
2. Selective fine tuning of CLIP encoders  
3. Scene graph based reasoning for explicit relationship modeling  
4. Ensemble based fusion strategies  
5. Targeted data augmentation for harder negative examples  

---

## Running the Code

1. Open `final v2.ipynb` in Google Colab  
2. Mount Google Drive when prompted  
3. Run all cells in order (approximately 30 to 45 minutes)  
4. Pretrained Faster R CNN and CLIP models are downloaded automatically  
5. Results and checkpoints are saved to Google Drive  

**Requirements:**

- Google Colab  
- Approximately 8 GB GPU memory  
- Stable internet connection  

---

## References

- StoryReasoning dataset by daniel3303  
- Faster R CNN from torchvision  
- CLIP ViT B 32 by OpenAI  
- PyTorch, HuggingFace Transformers, and Datasets  
