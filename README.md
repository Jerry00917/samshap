# Explain Any Concept: Segment Anything Meets Concept-Based Explanation (EAC) Poster @ NeurIPS 2023
Code for the paper "Explain Any Concept: Segment Anything Meets Concept-Based Explanation".


Here is an overview of our work, and you can find more in our [Preprint](https://arxiv.org/abs/2305.10289).
![Overview](./demo.png)

Our EAC approach generates high accurate and human-understandable post-hoc explanations.
![demo](./all_demo.png)

## Downloading the SAM backbone
We use ViT-H as our default SAM model. For downloading the pre-train model and installation dependencies, please refer [SAM repo](https://github.com/facebookresearch/segment-anything#model-checkpoints).

## Explain a hummingbird on your local pre-trained ResNet-50!
Simply run the following command:
```
python demo_samshap.py
```
