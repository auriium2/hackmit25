## Adapting Vision-Language Models for Neutrino Event Classification in High-Energy Physics

Dikshant Sagar ∗ 1 , Kaiwen Yu ∗ 1 , Alejandro Yankelevich 2 , Jianming Bian 2 , and Pierre Baldi 1

1 Department of Computer Science, University of California, Irvine, CA 92697

2 Department of Physics, University of California, Irvine, CA 92697

September 12, 2025

## Abstract

Recent advances in Large Language Models (LLMs)[1] have demonstrated their remarkable capacity to process and reason over structured and unstructured data modalities beyond natural language [2]. In this work, we explore the applications of Vision Language Models (VLMs), specifically a fine-tuned variant of LLaMa 3.2 [3] to the task of identifying neutrino interactions in pixelated detector data from high-energy physics (HEP) experiments. We benchmark these models against state-of-the-art convolutional neural network (CNN) architectures, similar to those used in the NOvA and DUNE experiments [4], [5], which have achieved high efficiency and purity in classifying electron and muon neutrino events. Our evaluation considers both the classification performance and interpretability of the model predictions. We find that VLMs can outperform CNNs, while also providing greater flexibility in integrating auxiliary textual or semantic information and offering more interpretable, reasoning-based predictions. This work highlights the potential of VLMs as a general-purpose backbone for physics event classification due to their high performance, interpretability, and generalizability, opening new avenues for integrating multimodal reasoning in experimental neutrino physics.

## 1 Introduction

Recent years have witnessed a surge in the adoption of machine learning across the physical sciences, driven by unprecedented volumes of experimental data and the promise of uncovering subtle patterns beyond the reach of traditional analyses.

* These authors contributed equally to this work

This is a preprint submitted to Nature Communications.

In high-energy physics (HEP), this trend is particularly evident: experiments generate vast streams of complex, high-dimensional detector outputs, making automated methods essential for transforming raw observations into scientifically meaningful insights[6], [7], [8], [9], [10], [11]. However, as the field increasingly turns to deep learning techniques, a persistent challenge remains: many of these models, while powerful, operate as opaque black boxes whose predictions are difficult to interpret and validate in a physics context [12].

Interestingly, this paradigm echoes the trajectory of computer vision research. For many years, computer vision depended on handcrafted feature extraction pipelines to identify salient characteristics in images. The advent of deep convolutional neural networks (CNNs) fundamentally changed this landscape by enabling models to learn hierarchical representations directly from raw pixel data, outperforming traditional methods and opening new frontiers in visual understanding[12], [14], [15]. Inspired by this progress, researchers in HEP have begun exploring deep learning architectures capable of processing detector data in similarly direct ways [9], [10], [11], [16].

A key example of this challenge arises in event classification, where the goal is to distinguish signal interactions of interest from a dominant background. For example, the ability to determine the flavor of neutrinos interacting in a detector is crucial for neutrino oscillation experiments, which aim to measure the rate at which neutrinos of certain flavors convert to different flavors along their trajectory between the source and detector. Historically, this event classification task has relied on first reconstructing higher-level objects within the detector, including resulting particle tracks and showers, and then summarizing their properties through a carefully selected set of engineered features [13]. These features, capturing energies, spatial configurations, and shape descriptors, have served as inputs to algorithms ranging from K-Nearest Neighbors and Boosted Decision Trees to shallow neural networks. While this approach has delivered strong results over decades of experimentation, it also has critical drawbacks: reconstruction errors can degrade classification performance, and the reliance on predefined features constrains the richness of information accessible to the model.

Most recently, Vision Language Models (VLMs), which are large neural networks pretrained on paired visual and textual data have emerged as a promising extension of these ideas. By jointly learning to associate image content with semantic information, these models can capture nuanced relationships and provide richer, more interpretable embeddings [17]. In the context of neutrino physics, where events can be represented as structured images or tensors and accompanied by labels or descriptions, VLMs offer an exciting opportunity to move beyond conventional pipelines. In addition to improving classification performance, these models can also generate natural-language explanations rooted in knowledge of the underlying physics processes, explicitly referencing event topologies such as muon tracks or electromagnetic showers, which help elucidate why a particular prediction was made and offer a path toward greater transparency and trust in machine learning-driven analyses.

In this work, we investigate fine-tuning VLMs for event classification in

high-energy physics neutrino experiments. Specifically, we consider this task in the context of a liquid argon time projection chamber (LArTPC), a relatively new particle detector technology known for its very high spatial and energy resolution. Our approach leverages the expressive capabilities of VLMs to extract features directly from low-level detector representations, reducing dependence on manually engineered variables. We show that with suitable adaptation, VLMs can deliver strong classification performance and offer new avenues for interpreting complex event signatures in neutrino detectors. In particular, we compare their performance against conventional CNNs and demonstrate that VLMs not only achieve superior classification accuracy but also provide a broader scope of reasoning and more informative explanations for their predictions. Finally, we demonstrate the ability of these VLMs to generalize beyond the specific datasets they are trained on and maintain high performance even under significantly degraded detector conditions, highlighting their robustness and adaptability. These results therefore suggest it would be possible to establish a reusable HEP foundation model, where future adaptations can be achieved even across experiments with minimal further fine-tuning.

## 2 Methods

## 2.1 Dataset

The dataset is a custom simulation of a modular LArTPC with square 5mm pixel-based readout. The detector is 2m × 2m × 7m in x, y, z with anodes at x = {-0 . 9m , -0 . 3m , 0 . 3m , 0 . 9m } and cathodes at x = {-0 . 6m , 0 . 0m , 0 . 6m } resulting in 0 . 3m drift lengths along x . Electron neutrino ( ν e ) and muon neutrino ( ν µ ) interactions are simulated with GENIE (v3.0.6) [18], [19] in the + z direction with uniform neutrino energy up to 10GeV . The dataset consists of 190,000 ν e and ν µ events, each with 74% of events interacting through the charged current and the rest through the neutral current, for which the neutrino flavor cannot be determined and is therefore a significant background for neutrino oscillation experiments. The energy deposition in liquid argon is then simulated with GEANT4 (v11.2.0) [20], [21]. To approximate the effect of drift electron transportation in liquid argon [22], [23], the energy deposition in each 1mm × 5mm × 5mm voxel is smeared with a Guassian filter of width 1 . 3mm ( 0 . 9mm ) in the transverse (longitudinal) direction per meter of drift distance to the anode. To generate images for training, two 2D event displays corresponding to XZ and YZ views are made by downsampling these voxels into 5cm × 5cm pixels to reduce the computational burden. Finally, we crop each event display to a 512 × 512 grayscale image ('pixel map') centered on the interaction, creating the final dataset for training.

Figure 1: LLaMa 3.2 Vision finetuning pipeline.

<!-- image -->

## 2.2 LLaMa 3.2 Vision

LLaMA Vision 3.2 is a suite of multimodal large language models developed by Meta, extending the LLaMA 3.2 series with visual capabilities [3]. Unlike traditional CNNs tailored specifically for image-based tasks, LLaMA Vision 3.2 integrates both textual and visual modalities within a unified transformer-based architecture[24]. It is trained on a diverse corpus of images and documents, enabling it to handle visual inputs such as photographs, rendered plots, and pixelated detector data alongside natural language. The model utilizes a highresolution vision encoder that tokenizes images into patch embeddings [25], which are then processed alongside text tokens by a shared transformer decoder. This allows for contextual reasoning across modalities, making the model well-suited for tasks that benefit from both visual understanding and symbolic reasoning, such as neutrino event classification in sparse detector images. In this work, we fine-tune the 11 billion parameter version of LLaMA Vision 3.2 using supervised instruction tuning and a parameter-efficient method known as QLoRA [26] on a labeled dataset of neutrino interaction pixel maps. This is visualized as a pipeline in Figure 1. This allows the model to learn physics-specific features while retaining its pretrained multimodal capabilities. One key advantage of this approach is the model's ability to produce not just classifications, but also textual justifications or descriptions of events, which can aid interpretability and experimental insight. By leveraging the flexibility and reasoning capabilities of LLaMA Vision 3.2, we aim to evaluate whether VLMs can serve as competitive or complementary alternatives to conventional CNN-based approaches in highenergy physics.

## 2.2.1 Parameter Efficient Supervised Finetuning

Fine-tuning large vision-language models like LLaMA Vision 3.2 [3] requires significant computational resources due to their billions of parameters and high

memory footprint. Fully fine-tuning all model weights is often infeasible, especially when working with domain-specific datasets that are relatively small and do not justify extensive retraining. Moreover, full fine-tuning can lead to overfitting and catastrophic forgetting of pretrained knowledge, particularly in specialized tasks such as neutrino interaction classification using sparse detector images [27]. To address these challenges, as shown in Figure 1, we adopt a parameter-efficient fine-tuning (PEFT) method, which enables task adaptation by training only a small subset of additional parameters while keeping the majority of the model frozen. Among various PEFT techniques, we employ QLoRA (Quantized Low-Rank Adaptation) [26] due to its memory efficiency, scalability, and strong empirical performance in both language and vision-language tasks. QLoRA combines two key ideas: (1) Quantization: The base model weights are stored in 4-bit precision, drastically reducing memory usage without significantly impacting performance. (2) Low-Rank Adaptation (LoRA) [27]: Trainable low-rank matrices are injected into the attention and MLP modules, enabling effective task-specific learning with a small number of parameters. By leveraging QLoRA [26], we are able to fine-tune LLaMA Vision 3.2 11B [3] on our neutrino dataset using modest GPU resources while preserving the general visual-linguistic reasoning capabilities of the original model. This approach enables faster iteration, reduced hardware demands, and easier experimentation, making it a practical strategy for applying large models in high energy physics contexts where computational resources may be constrained.

## 2.2.2 Model and Training Specifications

We fine-tune the LLaMA 3.2 Vision Instruct 11B model, a state-of-the-art multimodal large language model developed by Meta [3]. This model combines a high-capacity transformer-based language decoder [24] with a vision transformer (ViT)-style encoder [25], enabling joint processing of pixel-level visual data and textual instructions. The Instruct variant is specifically optimized for instructionfollowing, allowing us to formulate our event classification task as a multimodal prompt-response problem.

Weuse the meta-llama/Llama-3.2-11B-Vision-Instruct checkpoint, loaded with 4-bit quantization via the BitsAndBytes library to reduce GPU memory usage. The quantization setup is as follows:

```
BitsAndBytesConfig( load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16 )
```

This configuration enables us to fine-tune the model on four NVIDIA A6000 GPUs (49GB VRAM each) with a batch size of 4 per device with a balanced distributed strategy. To make fine-tuning more feasible on large models with limited resources, we employ QLoRA [26], a parameter-efficient method that trains a small number of injected low-rank matrices while keeping the base model frozen. Our QLoRA configuration includes:

```
LoraConfig( lora_alpha=16, lora_dropout=0.05, r=8, bias="none", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="VISION_MODEL", )
```

We use Hugging Face's [28] SFTTrainer for supervised fine-tuning, with training hyperparameters optimized for stability and efficiency. The model was fine-tuned for a single epoch using a batch size of 4 per device and gradient accumulation of 2, resulting in an effective batch size of 8. We used the adamw\_torch\_fused optimizer with a constant learning rate of 2 × 10 -4 , a warmup ratio of 0.03, and a maximum gradient norm of 0.3 to ensure stable updates. bfloat16 precision with TF32 fallback was employed to balance performance and numerical stability. Model checkpoints were saved every 500 steps, and training logs were recorded every 10 steps using TensorBoard. Using the training dataset comprising 190,000 events, the training run was completed in approximately one week.

## 2.2.3 Inference

Figure 2: LLaMa 3.2 Vision Classification Inference Pipeline

<!-- image -->

For model evaluation, we performed inference on an independent heldout test set comprising 5% of the dataset samples (10,000 events). Each sample consists of a pair of 2D pixel map images representing orthogonal views in the zx and zy planes. The model was loaded from the base weights ( meta-llama/Llama-3.2-11B-Vision-Instruct ) and further initialized with custom adapters we finetuned with our dataset. During inference, the model received a standardized system message that provided physics-specific context and described the distinguishing features of each interaction class and a user message instructing it to classify each event as one of three categories: electron

neutrino charged current ( ν e CC), muon neutrino charged current ( ν µ CC), or neutral current (NC) interactions similar to the finetuning stage, as shown in Figure 2.

To quantify model confidence in each prediction, we computed a joint probability distribution over the three target classes. Specifically, after generating the output text, we extracted the logarithmic softmax normalized probabilities corresponding to the first token of each class label at the decoding position immediately following the fixed prompt prefix ('I classify the pixel maps as'). This decoding index is where the model begins emitting the class label itself. For each of the three classes ( ν e CC, ν µ CC, and NC), we retrieved the log-probabilities of their respective canonical start tokens at this position. These values reflect the model's relative preference for each class when it commits to generating the label.

LLMs or VLMs predict text by autoregressively generating one token at a time, conditioning each new token on the input prompt as well as all previously generated tokens. Given an input prompt and accompanying visual information, the model outputs a probability distribution over the vocabulary for each decoding step, selecting the most likely tokens sequentially [1], [2]. However, unconstrained generation can produce variable or verbose phrasing inconsistent with standardized class labels, complicating automated parsing and evaluation. To mitigate this, we applied phrasal constraints, which, under the hood, run a constrained beam search during decoding [29]. Specifically, we enforced that the output must begin with a fixed phrase, "I classify the pixel maps as," followed by a token sequence corresponding exclusively to one of the target class labels ( ν e CC, ν µ CC, or NC). This was implemented by specifying the constrained prefix as a sequence of token IDs, ensuring that the beam search decoding process could only proceed along paths consistent with the constraint. As a result, during evaluation, the model was compelled to emit predictions in a consistent, machinereadable format while still leveraging its full generative capacity to condition on the visual features and prompt. This approach reduces variability in output text, simplifies downstream confidence scoring, and improves reproducibility of the inference results.

To convert these log-probabilities into normalized class probabilities, we applied a temperature-scaled softmax transformation. Concretely, the vector of log-probabilities was scaled by a scalar T to sharpen the distribution before applying the softmax function across the three classes:

<!-- formula-not-decoded -->

where P ( C i ) denotes the final confidence assigned to class C i and T = 5 . This procedure yields an interpretable probability distribution over the three classes for each prediction, emphasizing the most likely class while retaining information about the relative likelihoods of the alternatives [30], [31].

<!-- image -->

Z

Z

Figure 3: LLaMa 3.2 Vision Prediction Explanation.

## 2.2.4 Prediction Explainability

In neutrino physics, particularly in the classification of neutrino interaction events from detector pixel maps, interpretability is critical for validating model predictions against established physical understanding. A notable advantage of VLMs over conventional CNNs lies in their ability to provide human-readable explanations for their predictions. While CNNs primarily output numerical class probabilities or embeddings, their internal decision-making process is opaque, typically requiring post-hoc interpretability tools such as Grad-CAM, saliency maps, or feature visualization to approximate the reasoning behind a prediction. These methods can highlight regions of interest in the input image but do not inherently articulate why those regions influence the output [12], [32], [33].

Consequently, the explanation is grounded in both the visual patterns of the pixel maps and the physics concepts relevant to event topology. While these explanations are not a perfect reflection of the model's internal causal reasoning, they provide a more accessible and physics-aware interpretability interface than purely visual attribution maps from CNNs. This makes VLMs a promising direction for explainable AI in neutrino event classification.

In contrast, VLMs by virtue of their joint vision-language training, can generate natural language rationales that connect visual evidence to semantic concepts. Given an input image and a query, a VLM can not only identify the relevant object or scene but also explain its decision in textual form, often referencing specific visual cues [34]. For example, a VLM might classify an event as a 'muon neutrino charged-current interaction' and generate a textual explanation for detector pixel maps as shown in Figure 3.

## 2.3 CNN

Convolutional neural networks (CNNs) have long been widely used in problems such as event classification, feature extraction, and image segmentation in highenergy physics image analysis tasks[6], [16], [35]. Due to their advantages in local feature modeling and spatial invariance, CNN architectures exhibit good performance in processing sparse pixel maps, detector images, and other visual data[9], [36]. However, the expressive power of CNN models is often limited to the visual domain itself and lacks the ability to interpret information at the physical-semantic or symbolic level, which can be a limitation in scientific tasks that require incorporating contextual understanding or providing interpretable output[37]. Therefore, we use CNN as a comparative benchmark in this work to

explore its performance on neutrino image data and systematically compare it with our proposed multimodal macromodel, LLaMA Vision 3.2, to assess the latter's potential and advantages in combining visual understanding with textual inference.

## 2.3.1 Siamese CNN Architecture

We implement a lightweight convolutional neural network which adopts a Siamese architecture[38], [39], where a pair of input images are processed independently through identical sub-networks and later merged for joint reasoning. This convolutional baseline contains approximately 3.4 million parameters and is designed to efficiently handle high-resolution, sparsely populated pixel maps. It serves as a representative conventional CNN approach for neutrino event classification, enabling a direct comparison with the vision-language capabilities of LLaMA Vision 3.2.

After processing, the two feature branches are concatenated along the channel dimension and passed through additional inverted residual blocks[41]. The merged features are globally average pooled and flattened, followed by fully connected layers and dropout[43] for regularization. The model outputs a 3-class prediction corresponding to different types of neutrino interactions. The pipeline is given as Figure 4.

Each sub-network begins with a convolutional block using a ReLU6 activation[40], followed by a modified inverted residual block inspired by MobileNetV2[41]. The main body of the model is composed of several stacked inverted residual blocks with varying expansion factors, output channels, and kernel sizes. Certain blocks incorporate Squeeze-and-Excitation (SE) modules[42] to enhance channel-wise feature recalibration. Nonlinearities used include both ReLU6 and hard-swish, implemented via custom activation functions.

Figure 4: CNN Pipeline

<!-- image -->

## 2.3.2 Training Setup

We train the convolutional baseline model using a supervised learning framework. The input to the network consists of 2 grayscale detector images with a resolution of 512 × 512 . Each training sample consists of a pair of images corresponding

to a neutrino interaction event, processed through a Siamese architecture as described earlier.

The model is optimized using the Adam optimizer[44] with an initial learning rate of 1 × 10 -6 . A batch size of 16 is used during training. The model is trained for up to 300 epochs, with early stopping applied if the validation loss does not improve for 10 consecutive epochs. The objective function is the cross-entropy loss, and all training is performed on a single NVIDIA A5000 GPU using PyTorch. During the training, the program takes around 26 GB of memory, and takes around 210 minutes for one epoch of training.

## 3 Results

The LLaMa 3.2 Vision-Instruct model demonstrated strong performance in the classification of neutrino interaction events from simulated detector pixel maps. Compared to a conventional CNN baseline, our fine-tuned LLaMa 3.2 Vision consistently achieved higher accuracy, precision, recall, and AUC-ROC and more reliable confidence estimates.

Figures 5 and 6 show the confusion matrices of the LLaMa 3.2 Vision and CNN models, to evaluate and compare the classification between the two models. LLaMa 3.2 Vision demonstrates better classification performance across all classes. Its performance in NC identification is especially improved with better ν e vs NC discrimination, which is particularly important for neutrino oscillation experiments. LLaMa 3.2 Vision also demonstrates better balance both between efficiency (recall) and purity (precision), and among the three classes, leading to more reliable classification behavior.

Table 1 compares the classification performance and inference efficiency of the models. As shown, LLaMa 3.2 Vision achieves significantly higher classification metrics across the board, including an accuracy of 0.87, a precision and recall of 0.87, and an AUC-ROC score of 0.96. In contrast, the CNN baseline yields lower accuracy (0.68), as well as lower precision and recall (0.78 each), with a slightly reduced AUC-ROC of 0.93.

We also conducted generalization testing by running inference with both models on neutrino event pixel maps downsampled to half the original resolution (256 × 256). This setting evaluates each model's ability to maintain performance under reduced spatial detail, mimicking scenarios with lower detector resolution or aggressive data compression. As shown in Table 2, LLaMA 3.2 Vision consistently outperformed the CNN across all aggregated metrics. It achieved an accuracy, precision, and recall of 0.85, compared to 0.49 for the CNN, indicating a markedly higher rate of correct and balanced classifications. The difference was even more pronounced in AUC-ROC, with LLaMA 3.2 Vision scoring 0.95 versus the CNN's 0.72, demonstrating superior discriminative ability under distribution shift. We further present the confusion matrices (Figures 8 and 9) and the ROC curves (Figure 10) for this analysis. These results suggest that LLaMA 3.2 Vision's vision-language architecture is more robust to information loss and can better adapt to degraded visual inputs without retraining, a valuable property

for real-world neutrino event classification, where detector resolution and noise conditions can vary.

While the LLaMA 3.2 Vision model incurs substantially higher computational requirements, averaging 25.4 GB of memory (over 25× the 1.0 GB used by the CNN) and 3.3 seconds of inference time per sample, compared to just 20 milliseconds for the CNN. Its advantages extend beyond raw accuracy. In addition to achieving superior classification performance on neutrino event pixel maps, LLaMA 3.2 Vision offers an interpretability advantage that CNNs inherently lack. Leveraging its vision-language alignment, LLaMA 3.2 Vision can accompany its predictions with natural language explanations grounded in event topology, such as identifying long muon tracks, electromagnetic showers, or the absence of hadronic activity to justify its classification (Figure 3). This capability allows physicists to assess whether the model's reasoning is consistent with established physics heuristics, aiding both trust and error diagnosis. Furthermore, the additional resources required during training are justified by the potential to establish a reusable model that can be adapted to other applications with lightweight fine-tuning at a fraction of the effort. These results highlight a trade-off not merely between accuracy and efficiency, but between computational cost and the depth of insight and adaptability provided. While the CNN remains attractive for real-time or resource-constrained deployments due to its speed and lightweight footprint, LLaMA 3.2 Vision's richer output, combining high predictive power with human-readable justifications, makes it a compelling choice for offline analyses, detailed event studies, and contexts where explainability is as important as accuracy.

Table 1: Event classification aggregated metrics.

| Metric                      |   LLaMa 3.2 | Vision   |     CNN |
|-----------------------------|-------------|----------|---------|
| Accuracy                    |        0.87 |          |    0.68 |
| Precision                   |        0.87 |          |    0.78 |
| Recall                      |        0.87 |          |    0.78 |
| AUC-ROC                     |        0.96 |          |    0.93 |
| Inference Memory Usage (MB) |    25412.9  |          | 1000.57 |
| Time per Sample (mSec)      |     3300    |          |   20    |

Table 2: Event classification aggregated metrics for generalization testing.

| Metric    |   LLaMa 3.2 Vision |   CNN |
|-----------|--------------------|-------|
| Accuracy  |               0.85 |  0.49 |
| Precision |               0.85 |  0.49 |
| Recall    |               0.85 |  0.49 |
| AUC-ROC   |               0.95 |  0.72 |

Figure 5: Finetuned LLaMa 3.2 Vision's (a) recall matrix (truth normalized) and (b) precision matrix (prediction normalized).

<!-- image -->

Figure 6: CNN model's (a) recall matrix (truth normalized) and (b) precision matrix (prediction normalized).

<!-- image -->

Figure 7: ROC curves for each class (a) ν µ CC, (b) ν e CC, and (c) NC comparing performance between the finetuned LLaMa 3.2 Vision and the CNN.

<!-- image -->

Figure 8: Finetuned LLaMa 3.2 Vision's (a) recall matrix (truth normalized) and (b) precision matrix (prediction normalized) for generalization testing.

<!-- image -->

Figure 9: Siamese MobileNet model's (a) recall matrix (truth normalized) and (b) precision matrix (prediction normalized) for generalization testing.

<!-- image -->

Figure 10: ROC curves for each class (a) ν µ CC, (b) ν e CC, and (c) NC comparing performance between the finetuned LLaMa 3.2 Vision and the CNN for generalization testing.

<!-- image -->

## 4 Conclusion

We compared a CNN with the LLaMA 3.2 Vision model for neutrino event classification and observed a clear trade-off between computational efficiency and predictive capability. While LLaMA demands substantially higher computational resources, averaging 25.4 GB of memory usage and significantly longer inference times compared to the CNN. LLaMa 3.2 Vision consistently delivers superior accuracy across multiple evaluation settings, including challenging generalization scenarios with low resolution. Beyond raw performance, LLaMA 3.2 Vision offers the added advantage of interpretability through physics-grounded textual explanations, enabling the model to articulate reasoning tied to event topologies (e.g., identifying muon tracks, electromagnetic showers, or vertex activity). This capacity for explainable predictions is particularly valuable in scientific workflows, where transparent decision-making facilitates trust, debugging, and integration with expert knowledge.

These strengths make VLMs especially well-suited for offline analyses and detailed event studies in neutrino physics, where interpretability is as critical as accuracy. CNNs, on the other hand, retain an important role in scenarios requiring real-time inference or operation under strict resource constraints, such as on-detector edge computing or rapid online filtering. Looking ahead, promising research directions include compressing large models through quantization and pruning, distilling VLMs into compact architectures that retain interpretability, and developing domain-specific foundation models trained on diverse neutrino event topologies. Such efforts could bridge the gap between the accuracy and explainability of large-scale models and the efficiency of lightweight architectures, bringing the benefits of VLMs to a wider range of deployment environments in experimental physics.

## Acknowledgements

This work was supported by the U.S. Department of Energy under Award Number DE-SC0009920 awarded to J.B.

## References

- [1] Y. Chang et al., 'A survey on evaluation of large language models,' ACM transactions on intelligent systems and technology , vol. 15, no. 3, pp. 1-45, 2024.
- [2] J. Wu, W. Gan, Z. Chen, S. Wan, and P. S. Yu, 'Multimodal large language models: A survey,' in 2023 IEEE International Conference on Big Data (BigData) , IEEE, 2023, pp. 2247-2256.
- [3] A. Grattafiori et al., 'The llama 3 herd of models,' arXiv preprint arXiv:2407.21783 , 2024.
- [4] D. Ayres et al., 'The nova technical design report,' 2007.

- [5] A. Falcone, D. Collaboration, et al., 'Deep underground neutrino experiment: Dune,' Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment , vol. 1041, p. 167 217, 2022.
- [6] E. E. Robles, A. Yankelevich, W. Wu, J. Bian, and P. Baldi, 'Particle hit clustering and identification using point set transformers in liquid argon time projection chambers,' Journal of Instrumentation , vol. 20, no. 07, P07030, 2025.
- [7] A. Yankelevich, A. Shmakov, J. Bian, and P. Baldi, 'Sparse convolution transformers for dune fd event and particle classification,' Bulletin of the American Physical Society , 2024.
- [8] M. J. Fenton et al., 'Reconstruction of unstable heavy particles using deep symmetry-preserving attention networks,' Communications Physics , vol. 7, no. 1, p. 139, 2024.
- [9] P. Baldi, P. Sadowski, and D. Whiteson, 'Searching for exotic particles in high-energy physics with deep learning,' Nature communications , vol. 5, no. 1, p. 4308, 2014.
- [10] P. Baldi, K. Bauer, C. Eng, P. Sadowski, and D. Whiteson, 'Jet substructure classification in high-energy physics with deep neural networks,' Physical Review D , vol. 93, no. 9, p. 094 034, 2016.
- [11] P. Baldi, K. Cranmer, T. Faucett, P. Sadowski, and D. Whiteson, 'Parameterized neural networks for high-energy physics,' The European Physical Journal C , vol. 76, no. 5, pp. 1-7, 2016.
- [12] P. Baldi, Deep learning in science . Cambridge University Press, 2021.
- [13] C. Backhouse and R. Patterson, 'Library event matching event classification algorithm for electron neutrino interactions in the no ν a detectors,' Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment , vol. 778, pp. 31-39, 2015.
- [14] B. Abi et al., 'Neutrino interaction classification with a convolutional neural network in the dune far detector,' Physical Review D , vol. 102, no. 9, p. 092 003, 2020.
- [15] Y. LeCun and Y. Bengio, 'Convolutional networks for images, speech, and time series,' The handbook of brain theory and neural networks , 1998.
- [16] A. Aurisano et al., 'A convolutional neural network neutrino event classifier,' Journal of Instrumentation , vol. 11, no. 09, P09001, 2016.
- [17] J. Zhang, J. Huang, S. Jin, and S. Lu, 'Vision-language models for vision tasks: A survey,' IEEE transactions on pattern analysis and machine intelligence , vol. 46, no. 8, pp. 5625-5644, 2024.
- [18] C. Andreopoulos et al., 'The GENIE Neutrino Monte Carlo Generator,' Nucl. Instrum. Meth. A , vol. 614, pp. 87-104, 2010. doi : 10.1016/j.nima. 2009.12.009 arXiv: 0905.2517 [hep-ph] .

- [19] C. Andreopoulos et al., The GENIE Neutrino Monte Carlo Generator: Physics and User Manual , Oct. 2015. arXiv: 1510.05494 [hep-ph] .
- [20] Geant4 Collaboration, 'Geant4 10.4 release notes,' geant4-data.web.cern.ch , 2017. [Online]. Available: https://geant4-data.web.cern.ch/ReleaseNotes/ ReleaseNotes4.10.4.html
- [21] S. Agostinelli et al., 'GEANT4-a simulation toolkit,' Nucl. Instrum. Meth. A , vol. 506, pp. 250-303, 2003. doi : 10.1016/S0168-9002(03)01368-8
- [22] Liquid argon properties (tables and calculators) . [Online]. Available: https: //lar.bnl.gov/properties/
- [23] Y. Li et al., 'Measurement of longitudinal electron diffusion in liquid argon,' "Nucl. Instrum. Meth. A" , vol. 816, pp. 160-170, 2016, issn : 01689002. doi : https://doi.org/10.1016/j.nima.2016.01.094 [Online]. Available: https://www.sciencedirect.com/science/article/pii/ S0168900216001443
- [24] A. Vaswani et al., 'Attention is all you need,' Advances in neural information processing systems , vol. 30, 2017.
- [25] A. Dosovitskiy et al., 'An image is worth 16x16 words: Transformers for image recognition at scale,' arXiv preprint arXiv:2010.11929 , 2020.
- [26] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, 'Qlora: Efficient finetuning of quantized llms,' Advances in neural information processing systems , vol. 36, pp. 10 088-10 115, 2023.
- [27] E. J. Hu et al., 'Lora: Low-rank adaptation of large language models.,' ICLR , vol. 1, no. 2, p. 3, 2022.
- [28] T. Wolf et al., 'Huggingface's transformers: State-of-the-art natural language processing,' arXiv preprint arXiv:1910.03771 , 2019.
- [29] C. Hokamp and Q. Liu, 'Lexically constrained decoding for sequence generation using grid beam search,' arXiv preprint arXiv:1704.07138 , 2017.
- [30] F. Petroni et al., 'Language models as knowledge bases?' arXiv preprint arXiv:1909.01066 , 2019.
- [31] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, 'On calibration of modern neural networks,' in International conference on machine learning , PMLR, 2017, pp. 1321-1330.
- [32] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, 'Grad-cam: Visual explanations from deep networks via gradientbased localization,' in Proceedings of the IEEE international conference on computer vision , 2017, pp. 618-626.
- [33] M. Sundararajan, A. Taly, and Q. Yan, 'Axiomatic attribution for deep networks,' in International conference on machine learning , PMLR, 2017, pp. 3319-3328.

- [34] F. Sammani, T. Mukherjee, and N. Deligiannis, 'Nlx-gpt: A model for natural language explanations in vision and vision-language tasks,' in proceedings of the IEEE/CVF conference on computer vision and pattern recognition , 2022, pp. 8322-8332.
- [35] Madrazo, Celia Fernández, Heredia, Ignacio, Lloret, Lara, and Marco de Lucas, Jesús, 'Application of a convolutional neural network for image classification for the analysis of collisions in high energy physics,' EPJ Web Conf. , vol. 214, p. 06 017, 2019. doi : 10.1051/epjconf/201921406017 [Online]. Available: https://doi.org/10.1051/epjconf/201921406017
- [36] R. Acciarri et al., 'Convolutional neural networks applied to neutrino events in a liquid argon time projection chamber,' Journal of instrumentation , vol. 12, no. 03, P03011, 2017.
- [37] G. Carleo et al., 'Machine learning and the physical sciences,' Reviews of Modern Physics , vol. 91, no. 4, p. 045 002, 2019.
- [38] J. Bromley, I. Guyon, Y. LeCun, E. Säckinger, and R. Shah, 'Signature verification using a" siamese" time delay neural network,' Advances in neural information processing systems , vol. 6, 1993.
- [39] G. Koch, R. Zemel, R. Salakhutdinov, et al., 'Siamese neural networks for one-shot image recognition,' in ICML deep learning workshop , Lille, vol. 2, 2015, pp. 1-30.
- [40] A. F. Agarap, 'Deep learning using rectified linear units (relu),' arXiv preprint arXiv:1803.08375 , 2018.
- [41] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, 'Mobilenetv2: Inverted residuals and linear bottlenecks,' in Proceedings of the IEEE conference on computer vision and pattern recognition , 2018, pp. 4510-4520.
- [42] J. Hu, L. Shen, and G. Sun, 'Squeeze-and-excitation networks,' in Proceedings of the IEEE conference on computer vision and pattern recognition , 2018, pp. 7132-7141.
- [43] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, 'Dropout: A simple way to prevent neural networks from overfitting,' The journal of machine learning research , vol. 15, no. 1, pp. 1929-1958, 2014.
- [44] D. P. Kingma and J. Ba, 'Adam: A method for stochastic optimization,' arXiv preprint arXiv:1412.6980 , 2014.