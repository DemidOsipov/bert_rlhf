# Reinforcement Learning from Human Feedback (RLHF) Implementation

This repository contains a comprehensive implementation of RLHF techniques for fine-tuning language models. The project demonstrates two distinct applications of RLHF: sentiment control and response length optimization.

## Overview

The implementation follows the standard RLHF pipeline consisting of three main stages:
1. **Reward Model Training**: Training a classifier to predict human preferences
2. **Policy Optimization**: Fine-tuning the language model using reinforcement learning
3. **Evaluation**: Assessing the effectiveness of the alignment process

## Project Structure

The notebook is organized into two main experimental sections:

### Part 1: Sentiment Control
Training GPT-2 to generate negative movie reviews using the IMDB dataset. This section demonstrates how RLHF can be used to control the sentiment of generated text.

### Part 2: Response Length Optimization
Fine-tuning GPT-2 Large to generate shorter responses by using negative response length as a reward.

## Dependencies

The implementation requires the following libraries:
- `transformers==4.33.1` - Hugging Face transformers library
- `trl==0.7.4` - Transformer Reinforcement Learning library
- `accelerate==0.28.0` - Distributed training acceleration
- `peft==0.5.0` - Parameter Efficient Fine-Tuning
- `datasets` - Dataset loading and processing
- `torch` - PyTorch deep learning framework
- `numpy` - Numerical computations
- `matplotlib` - Visualization
- `tqdm` - Progress bars

## Algorithms and Techniques

### Reward Model Training
- **Architecture**: DistilBERT-base-cased for Part 1, DeBERTa-v3-base for Part 2
- **Training Objective**: Pairwise ranking loss following the InstructGPT methodology
- **Dataset**: IMDB movie reviews (Part 1), Synthetic-instruct-gptj-pairwise (Part 2)

### Reinforcement Learning
- **Algorithm**: Proximal Policy Optimization (PPO)
- **Policy Model**: GPT-2 (Part 1), GPT-2 Large (Part 2)
- **Parameter Efficiency**: LoRA (Low-Rank Adaptation) for efficient fine-tuning

### Key Implementation Details
- **Gradient Checkpointing**: Memory optimization during training
- **Mixed Precision Training**: FP16 for faster computation
- **KL Divergence Regularization**: Prevents excessive deviation from base model
- **Adaptive Learning Rate**: Dynamic adjustment based on KL divergence

## Methodology

### Reward Model
The reward models are trained using a pairwise preference learning approach. For sentiment control, the model learns to assign higher scores to negative reviews. For length optimization, the reward is simply the negative word count of the generated text.

### Policy Optimization Process
1. **Rollout Phase**: Generate responses using the current policy
2. **Evaluation Phase**: Compute rewards using the trained reward model
3. **Update Phase**: Apply PPO updates to maximize expected reward

### Evaluation Metrics
- **Reward Model Accuracy**: Percentage of correct pairwise comparisons
- **Average Reward**: Mean reward across generated samples
- **KL Divergence**: Measure of deviation from the original model
- **Response Length Distribution**: Histogram analysis of generated text lengths

## Results

### Part 1: Sentiment Control
- Reward model achieves 97.7% accuracy on train set and 97.3% on test set
- Successfully fine-tuned GPT-2 to generate predominantly negative movie reviews
- Preserved text quality

### Part 2: Length Optimization
- Reward model reaches 99.9% accuracy on preference classification task
- **Significant reduction in average response length after RLHF training**
- Preserved text quality

## Technical Considerations

### Memory Management
- Gradient checkpointing reduces memory usage at the cost of ~30% slower training
- LoRA adapters significantly reduce the number of trainable parameters
- Batch size adjustments may be necessary based on available hardware

## Future Extensions

This implementation can be used to shift LLM's output in any direction using reward model.
However, for most of the tasks it would require way more compute and fine-tuning.

## References

The implementation is based on established RLHF methodologies from recent literature, particularly the InstructGPT framework and PPO algorithm foundations. The TRL library provides the core infrastructure for efficient RLHF training.
