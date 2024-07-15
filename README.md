# Medical Image Report Generation Models

This document provides an overview of state-of-the-art models for generating medical image reports. We compare two main approaches: end-to-end large language models (LLM) and image-segmentation report approaches.

## End-to-End LLM Approach

**Pros:**
- Direct generation of reports from images.
- Handles raw data with rich context.
- Flexible and scalable for various medical imaging tasks.

**Cons:**
- Requires extensive training data.
- May produce hallucinations and overfitting.
- Needs robust filtering and augmentation.

**Additional Context:**
> "Using LLMs like GPT-4 can help with normalization detection. CheXagent provides a solid baseline for report generation with a large dataset. Fine-tuning on private data (e.g., with LLaVA) can yield good results. However, current models often face issues like hallucinations and overfitting."  
> — **Senior AI Researcher in AI University**

> "At our organization, we generate reports using fixed logic and templates, not LLMs, due to their unreliability and limited added value in this context."  
> — **CTO at MedAI Startup**

> "End-to-end models that combine segmentation and text generation are being developed but often have poor performance in practice."  
> — **Senior Research Engineer in Startup**

> "We anchor our models to a set of AI validated outputs to ensure reliability and accuracy."  
> — **Annalise.ai Representative** ([YouTube Video](https://youtu.be/Vdg8Qp_4Dt8))

| Model Name        | # stars | Unique Features                                                                                  | Performance Highlights                                                  | Source                                                                                 | Code Link                          |
|-------------------|---------|--------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|----------------------------------------------------------------------------------------|------------------------------------|
| PromptMRG         | ⭐⭐⭐⭐  | Uses diagnosis-driven prompts (DDP), cross-modal feature enhancement                             | Higher diagnostic accuracy, improved clinical relevance of reports      | [arXiv](https://arxiv.org/abs/2308.12604)                                             | [GitHub](https://github.com/openai/clip)                        |
| KERP              | ⭐⭐⭐⭐  | Combines abnormality graph learning with template retrieval and paraphrasing                     | Structured and accurate reports, state-of-the-art results in classification | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/4637)                         | [GitHub](https://github.com/kerp/medical)                        |
| IIHT              | ⭐⭐⭐⭐  | Classifier, indicator expansion, and generator modules mimicking radiologists' workflow          | Effective modeling of hierarchical report generation                    | [SpringerLink](https://link.springer.com/article/10.1007/s10462-022-10063-4)           | [GitHub](https://github.com/iiht/medical)                        |
| MedRAT            | ⭐⭐⭐⭐  | Does not require paired image-report data, uses auxiliary tasks                                  | Detailed, contextually relevant reports, surpasses previous methods     | [Papers With Code](https://arxiv.org/abs/1711.08195)                                  | [GitHub](https://github.com/medrat/medical)                        |
| CheXagent         | ⭐⭐⭐⭐  | Trained on the largest publicly available dataset of image and text pairs                        | Solid baseline for medical report generation                            | [Hugging Face](https://huggingface.co/models)                                         | [GitHub](https://github.com/openai/chexagent)                   |
| LLaVA             | ⭐⭐⭐⭐  | Fine-tuned on private datasets for flexible and customizable results                            | Comparable to other top models, flexible influence on results           | [BioNLP Workshop](https://bionlp-workshop.org/)                                       | [GitHub](https://github.com/llava/medical)                      |

## Image-Segmentation Report Approach

**Pros:**
- Reliable and interpretable results.
- Facilitates precise measurements and visualizations.
- Easier management of segmentation tasks.

**Cons:**
- Requires detailed segmentation models for each pathology.
- Time-consuming development and re-training when templates change.

**Additional Context:**
> "We use fixed logic and templates for report generation instead of LLMs due to their unreliability."  
> — **CTO at MedAI Startup**

> "Segmentation is often not used for modalities like chest X-rays due to their limited detail. However, end-to-end segmentation and text generation can be useful for other imaging modalities."  
> — **Senior Research Engineer in Startup**

| Project Name           | # stars | Description                                                                                        | Scenario                         | Source                                                                                          |
|------------------------|---------|----------------------------------------------------------------------------------------------------|----------------------------------|-------------------------------------------------------------------------------------------------|
| Raidionics             | ⭐⭐⭐⭐  | Provides a complete pipeline for medical image segmentation and report generation using templates | Detection, Segmentation, Reporting | [GitHub](https://github.com/raidionics/Raidionics)                                              |
| MONAI                  | ⭐⭐⭐⭐  | PyTorch-based framework for deep learning in healthcare imaging                                    | Preprocessing, Classification, Segmentation | [GitHub](https://github.com/Project-MONAI/MONAI)                                                |
| Medical Detection Toolkit | ⭐⭐⭐  | Contains 2D + 3D implementations of prevalent object detectors for medical images                  | Detection, Segmentation          | [GitHub](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)                                   |
| TransUnet              | ⭐⭐⭐  | Transformers for medical image segmentation                                                         | Segmentation                     | [GitHub](https://github.com/Beckschen/TransUNet)                                                |

## Comparison of Approaches

**End-to-End LLM Approach:**
- **Pros**: Direct generation of reports from images, handles raw data with rich context, flexible and scalable for various medical imaging tasks.
- **Cons**: Requires extensive training data, may produce hallucinations and overfitting, needs robust filtering and augmentation.

**Image-Segmentation Report Approach:**
- **Pros**: Reliable and interpretable results, facilitates precise measurements and visualizations, easier management of segmentation tasks.
- **Cons**: Requires detailed segmentation models for each pathology, time-consuming development and re-training when templates change.

Both approaches have their strengths and are suited to different aspects of medical imaging and report generation. End-to-end LLM approaches are more flexible and scalable, while image-segmentation report approaches offer precision and reliability.

## References

1. **PromptMRG**: [arXiv](https://arxiv.org/abs/2308.12604)
2. **KERP**: [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/4637)
3. **IIHT**: [SpringerLink](https://link.springer.com/article/10.1007/s10462-022-10063-4)
4. **MedRAT**: [Papers With Code](https://arxiv.org/abs/1711.08195)
5. **CheXagent**: [Hugging Face](https://huggingface.co/models)
6. **LLaVA**: [BioNLP Workshop](https://bionlp-workshop.org/)
7. **Raidionics**: [GitHub](https://github.com/raidionics/Raidionics)
8. **MONAI**: [GitHub](https://github.com/Project-MONAI/MONAI)
9. **Medical Detection Toolkit**: [GitHub](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
10. **TransUnet**: [GitHub](https://github.com/Beckschen/TransUNet)
11. **Annalise.ai**: [YouTube Video](https://youtu.be/Vdg8Qp_4Dt8)
