# CHAPTER 1 INTRODUCTION
## 1.1 Introduction
KneeAssist AI is a local research application that classifies a complete knee magnetic resonance imaging (MRI) study as normal or abnormal. The general-abnormality output is the primary academic task. ACL tear and meniscal tear scores are retained as secondary research outputs to show how the system can be extended when suitable labels are available. This report documents data preparation, model comparison, evaluation, explanation and the user interface. Its central question is whether a reproducible study-level normal/abnormal classifier can provide a transparent basis for research testing while making the limits of its predictions visible.

The implemented system is technically operational, but its measured performance does not justify routine clinical deployment. The selected model achieved a macro area under the receiver operating characteristic curve (AUROC) of 0.8267 on 120 MRNet official validation studies. External-reference AUROCs were 0.6030 for ACL injury and 0.6561 for meniscal annotations. These results support further investigation rather than a claim of reliable diagnosis. All numerical project results in this report refer to the preserved release evidence verified on 27 September 2026.

## 1.2 Background and context
An MRI examination contains related slices rather than one independent photograph. A classifier that receives isolated screenshots may lose anatomical context, sequence information and the relationship between planes. This project therefore treats a study as its input and combines available axial, coronal and sagittal stacks. Public knee MRI research provides a suitable development foundation: MRNet specifically investigated abnormality, ACL tears and meniscal tears using retrospective MRI data (Bien et al., 2018).

The project is situated within data science and informatics: it joins data provenance, predictive modelling, software engineering and evidence reporting. Its potential relevance to Zimbabwe is educational and investigational. No Zimbabwean hospital dataset, local prevalence estimate, reader study or local clinical workflow evaluation was available. International benchmark performance must therefore not be presented as evidence of effectiveness in Zimbabwean practice.

### 1.2.1 Current knowledge
Residual networks provide a well-established image representation architecture, while EfficientNet explores systematic scaling of network dimensions (He et al., 2016; Tan & Le, 2019). In this project these architectures serve as pretrained slice encoders. Neither architecture alone determines whether a study-level system will generalise to a different institution or acquisition protocol.

Medical imaging evaluation also depends on how data are separated, how references are defined and what the reported metric measures. A methodological review identifies weaknesses in evaluation practices that can undermine apparent progress (Varoquaux & Cheplygina, 2022). This motivates explicit distinctions between fitting, model selection, calibration and evaluation in the present work.

### 1.2.2 Research gap
The practical gap addressed here is an auditable local knee MRI workflow that connects normal/abnormal study prediction, optional secondary label outputs, transparent data splits, comparative model selection and usable results presentation. The project does not claim to introduce a new neural architecture or solve clinical generalisation. Its contribution is the implemented and documented combination of these components, including evidence that acceptable internal ranking can coexist with weak transfer to other datasets.

## 1.3 Problem statement
A knee MRI classifier can produce plausible numerical outputs without establishing that its inputs are appropriate, its evaluation is independent or its findings are clinically dependable. For the available project data, study identifiers and pathology labels support model development, but patient linkage, a new untouched test cohort and local clinical validation are absent. The problem is therefore to construct and evaluate a reproducible research system for three knee findings while exposing these limitations and preventing unsupported claims about diagnostic performance.

## 1.4 Aim
The aim is to develop and evaluate a reproducible deep-learning system for study-level classification of normal and abnormal knee MRI studies, with transparent evidence, visual explanations and a local user interface.

## 1.5 Specific objectives
1. To prepare and audit knee MRI studies and normal/abnormal reference labels using reproducible preprocessing and study-level leakage checks.
2. To develop and compare ResNet-18 and EfficientNet-B0 models for study-level normal and abnormal knee MRI classification.
3. To evaluate and calibrate the selected normal/abnormal model using defined MRNet partitions, reporting performance and uncertainty.
4. To implement an independent MRI inference pipeline with Grad-CAM visualisations and traceable normal/abnormal results summaries.
5. To develop and verify a local Streamlit interface for MRI upload, slice review, classification display, export and case clearing.

## 1.6 Motivation
The technical motivation is to move beyond an isolated training notebook toward a research application whose predictions can be traced to a dataset role, preprocessing configuration and checkpoint. A second motivation is to learn from model failures rather than conceal them. The observed external performance makes the distinction between software completion and clinical suitability central to the project. No personal experience, clinical placement or consultation is claimed on behalf of the student.

## 1.7 Stakeholders
Table 1.1 Stakeholders and their relationship to the prototype
| Stakeholder | Relevant use | Boundary |
| Student researcher | Reproduce experiments and explain results | Responsible for understanding and reviewing this report |
| Supervisor and examiners | Assess objectives, evidence and scholarship | Approval and assessment are not assumed |
| Imaging researchers | Inspect study predictions and model attention | Research use with suitable data governance |
| Clinicians | Inform future requirements and evaluation design | No validated clinical deployment is claimed |
| Dataset custodians | Specify access and permitted use | Original data and private links are not redistributed |

## 1.8 Scope
The implemented core scope consists of normal/abnormal classification, two pretrained encoder families, study-level slice aggregation, MRNet-only fitting, reserved calibration, external-reference evaluation and a local Streamlit application. It includes visualisation of MRI slices and Grad-CAM attention, checkpoint identification and export of a structured summary. ACL and meniscal outputs remain secondary research functions and are not the primary claim in the title. Supported inputs are prepared MRI NumPy arrays, a ZIP containing named arrays, or a guarded DICOM ZIP containing one readable series in each explicitly labelled plane folder.

The scope excludes diagnosis from internet pictures, JPEG/PNG screenshots, arbitrary multi-series DICOM routing, clinically validated NIfTI/DICOM deployment, treatment recommendations, fracture detection and comprehensive assessment of all knee disease. It also excludes prospective clinical evaluation and automated confirmation that an upload contains a knee or belongs to one patient. Removing picture input avoids suggesting that a classifier developed for MRI stacks has been validated for arbitrary images.

## 1.9 Assumptions
The experiments assume that public source labels and retained study identifiers have been interpreted correctly. This is an operational assumption, not independent verification of every radiological label. Input arrays are assumed to represent appropriately prepared MRI stacks. Study separation and exact-file comparisons are assumed to reduce identified leakage routes; they cannot prove patient-level independence when patient mappings are missing. Hardware-specific numerical behaviour is treated as something to test rather than assume.

## 1.10 Relevance and significance
### 1.10.1 Relevance
The work demonstrates an integrated data science process: define a prediction task, document source limitations, fit competing models, select on a designated development partition and report failures on other cohorts. These practices are directly relevant to reproducible informatics education. They are also useful for planning a future local validation study without confusing public-data experiments with deployment evidence.

### 1.10.2 Significance
The principal result is a working, inspectable research system accompanied by a negative generalisation finding. Preserving a selected checkpoint despite disappointing external results prevents selective reporting from creating a misleading success story. The system can support teaching, further experiments and requirements discussions. Benefits to patient outcomes, reporting time or healthcare costs remain unmeasured.

## 1.11 Chapter summary
This chapter defined the problem, scope and five objectives. The remaining chapters review the supporting literature, describe the implemented methods, present measured outcomes and identify the work required before stronger claims could be made.

# CHAPTER 2 LITERATURE REVIEW
## 2.1 Introduction
This chapter uses a targeted narrative review to connect knee MRI classification with model design, reliable evaluation and explainability. Twenty-one refereed journal or conference publications were retained. Foundational work is included because the implemented encoders, optimisation and evaluation methods depend on it. The review distinguishes direct knee imaging evidence from general machine-learning methods and reporting guidance.

## 2.2 Research questions
Table 2.1 Alignment of research questions and objectives
| Objective | Research question | Evidence needed |
| 1 | Can the available studies be prepared with traceable labels and defensible separation? | Manifests, source roles and leakage audit |
| 2 | Which candidate ranks highest on the designated internal selection partition? | Comparable internal AUROC and saved checkpoints |
| 3 | How well do its scores discriminate and classify on specified evaluation cohorts? | Per-target metrics, calibration description and uncertainty |
| 4 | Can a saved model independently generate inspectable, traceable output? | Real-case inference, attention output and checkpoint identity |
| 5 | Can the local application complete the intended researcher workflow? | Upload, analysis, visualisation, export and clear verification |

The questions separate predictive quality from engineering functionality. A successful upload does not answer the discrimination question, and a high internal AUROC does not establish that the interface is safe or usable in a clinic. Each question has an observable answer within the present project, even where the answer exposes a limitation.

## 2.3 Search strategy and data sources
The documentation search was conducted through web search, publisher pages, official conference proceedings, PubMed/PMC records and DOI metadata. Search concepts included knee MRI classification, MRNet, KneeMRI ACL injury, fastMRI pathology annotations, residual networks, EfficientNet, calibration, leakage, external validation, saliency maps and prediction-model reporting. Reference metadata were checked against primary publication records or DOI records. The retained-source register appears in Appendix B.

This was not a systematic review. No claim is made that Scopus or Web of Science was searched, that every eligible study was retrieved, or that duplicate screening and independent reviewers were used. Search totals and a PRISMA flow diagram are not fabricated. The selection favoured direct dataset sources, original methods used in the implementation and publications explaining threats to interpretation. Sources without a clear relationship to the implemented task were not added merely to inflate the bibliography.

## 2.4 Data extraction
For each retained publication, the extraction recorded author and year, publication identity, study or method type, relevance to the project, and a transfer limitation. Direct knee MRI sources inform dataset and task interpretation. Architecture and optimisation papers inform implementation choices. Evaluation and reporting papers inform the strength of permissible conclusions. This separation avoids treating an algorithm paper as proof of clinical effectiveness.

### 2.4.1 Direct evidence on knee MRI
Bien et al. (2018) provide the closest task match: retrospective knee MRI classification for abnormality, ACL and meniscal tears. The present project uses the same broad target names but does not reproduce the original training architecture or evaluation protocol exactly. Published MRNet performance is therefore contextual evidence rather than a directly comparable control result.

Štajduhar et al. (2017) investigate ACL injury detection from MRI. Their work supports a separate ACL-focused reference evaluation, but an ACL label does not establish general knee normality or provide a meniscus label. Consequently, KneeMRI cannot be treated as a complete three-target external test for this application.

Zhao et al. (2022) describe fastMRI+ pathology annotations linked to fastMRI examinations. These annotations extend the value of a reconstruction-oriented resource for pathology research. In the local subset, absence of a matching meniscal annotation is a weaker reference than an independently confirmed normal meniscus. The report therefore uses the term external-reference evaluation and retains this uncertainty in interpretation.

### 2.4.2 Image encoders and regularisation
He et al. (2016) introduce residual learning, enabling deeper image networks through residual connections. ResNet-18 provides a comparatively compact baseline in the present study. Its value here is a standard, reproducible comparison architecture rather than an assumption that increasing depth necessarily improves knee MRI predictions.

Tan and Le (2019) present EfficientNet and compound scaling. EfficientNet-B0 is used as the second encoder because it offers a different accuracy–computation design. Its selection in this project follows measured internal AUROC. The original paper does not establish that EfficientNet-B0 is superior for this particular medical dataset.

Srivastava et al. (2014) describe dropout as a regularisation method that reduces co-adaptation during training. The local classification head includes dropout, but the experiment does not isolate its causal contribution. An ablation with other settings fixed would be required to establish whether the chosen dropout level improved these results.

Loshchilov and Hutter (2019) distinguish decoupled weight decay from the conventional coupling of regularisation with adaptive optimisation. AdamW is used in the project. Its presence makes the training procedure explicit; it does not eliminate the need to monitor overfitting or justify an unmeasured advantage over other optimisers.

Paszke et al. (2019) describe PyTorch as an imperative deep-learning framework. It supports the implementation of pretrained encoders, automatic differentiation and GPU execution. Framework capability must be distinguished from experiment quality: a correct tensor computation can still produce a biased estimate if the split or reference labels are inappropriate.

### 2.4.3 Evaluation and calibration
Steyerberg et al. (2010) distinguish different aspects of prediction-model performance, including discrimination and overall predictive error. This supports reporting several complementary measures. In the local study, AUROC is the selection criterion, while threshold-dependent metrics describe a particular operating point. Neither substitutes for evidence of clinical benefit.

Saito and Rehmsmeier (2015) explain why precision–recall plots can be informative for imbalanced binary tasks. The project therefore retains precision–recall curves alongside ROC curves. Their shapes must be read in the context of the positive proportion in the evaluation cohort rather than compared across datasets as if prevalence were irrelevant.

Youden (1950) proposes an index combining sensitivity and specificity. The implementation uses the maximum of sensitivity plus specificity minus one on the calibration partition to select research thresholds. This criterion gives equal statistical emphasis to the two rates; it does not encode the clinical costs of missed injury and unnecessary follow-up.

Guo et al. (2017) investigate calibration of modern neural networks. Their work motivates checking the relationship between model scores and observed outcomes. The local method is positive-slope Platt calibration, not a replication of their temperature-scaling experiment. Calibration estimated from a small development subset is not evidence of transportable clinical probability estimates.

Efron (1979) develops bootstrap resampling as a method for estimating sampling uncertainty. The project uses study resampling to obtain AUROC intervals. These intervals describe variation under the implemented resampling scheme; they omit variation from new training seeds, model selection and unavailable patient clustering.

### 2.4.4 Generalisation and reproducibility
Kapoor and Narayanan (2023) analyse leakage and reproducibility problems in machine-learning-based science. Their discussion motivates making dataset roles explicit and restricting fitted transformations to the fitting data. Exact-file checks are useful safeguards, but a passed file audit cannot establish absence of every form of dependence.

Varoquaux and Cheplygina (2022) discuss methodological failures in medical imaging machine learning. The present report responds by describing the reused validation cohort, one-seed design and incomplete patient linkage. Transparent reporting does not remove these weaknesses; it enables readers to judge their effect on the conclusions.

Zech et al. (2018) examine variable generalisation in pneumonia detection across chest-radiograph sources. This is not knee MRI evidence. Its relevance is methodological: strong performance in one source need not persist elsewhere. The local knee results provide the project-specific evidence of that concern.

### 2.4.5 Explanations and reporting
Selvaraju et al. (2017) introduce Grad-CAM, which uses gradients to produce class-related visual localisation. KneeAssist AI uses this form of attention visualisation to accompany study scores. A highlighted region is not treated as a segmented tear or verified anatomical lesion.

Adebayo et al. (2018) propose sanity checks for saliency methods. Their work cautions against interpreting visual plausibility alone as evidence that a heatmap explains the learned decision faithfully. This project verifies heatmap generation but has not completed model-randomisation or expert lesion-localisation validation.

Mongan et al. (2020) provide CLAIM reporting guidance for medical imaging AI. Collins et al. (2024) provide TRIPOD+AI guidance for prediction-model reporting. These are reporting resources, not approvals or certificates. They motivate explicit descriptions of data, intended use, model development and limitations, without implying that this report has undergone an independent checklist audit.

Kelly et al. (2019) discuss challenges in obtaining clinical impact from AI. This supports separating a functioning research application from a validated service. In this project, prospective usefulness, workflow impact and patient outcomes remain outside the evidence obtained.

## 2.5 Quality assessment
The review assessed whether each source had a verifiable publication identity, whether it directly addressed the relevant concept and whether its claims transferred to the current setting. Dataset papers were considered for target compatibility and reference construction; method papers for implementation relevance; and reporting papers for transparency requirements. This was a structured narrative appraisal, not a numerical risk-of-bias instrument or a meta-analysis.

PRISMA reporting guidance is not itself a risk-of-bias scoring tool, and this narrative review does not claim PRISMA compliance. Similarly, a refereed publication does not make every downstream use of its data valid. The most consequential local quality issue is that the three resources support different labels and evaluation units. A literature source can establish that a dataset exists while leaving the local subset and conversion pipeline to be audited separately.

Table 2.2 Evidence categories and transfer limits
| Category | Included sources | Main transfer limit |
| Knee MRI and annotations | Bien; Štajduhar; Zhao | Different labels, sampling and study units |
| Encoders and training | He; Tan and Le; Srivastava; Loshchilov and Hutter; Paszke | General methods do not establish local clinical accuracy |
| Metrics and uncertainty | Steyerberg; Saito and Rehmsmeier; Youden; Guo; Efron | Operating points and uncertainty depend on the protocol |
| Reliability and transfer | Kapoor and Narayanan; Varoquaux and Cheplygina; Zech | Cautions require local empirical investigation |
| Explanation and reporting | Selvaraju; Adebayo; Mongan; Collins; Kelly | Explainability and reporting are not clinical validation |

## 2.6 Concepts and themes
### 2.6.1 Prediction unit and dependence
The prediction unit is the complete study rather than a randomly selected slice. Slices from one volume are closely related, so allocating them independently across fitting and evaluation sets would answer an easier and less useful question. The project keeps study members together. A remaining issue is that different study identifiers could still belong to one patient; absent a linkage table, that dependence cannot be ruled out.

### 2.6.2 Label semantics
General abnormality, ACL tear and meniscal tear are distinct binary tasks. A study may be positive for more than one, which supports a multi-label output rather than mutually exclusive class selection. The external ACL resource cannot supply absent meniscal references. Likewise, a missing fastMRI annotation cannot automatically be promoted to an exhaustive normal diagnosis. Semantic compatibility is therefore more important than simply increasing the number of files.

### 2.6.3 Ranking and decisions
AUROC describes ranking across thresholds, while a displayed positive status depends on a selected threshold. Precision further depends on the positive proportion in the evaluated population. A model can have useful ranking yet generate many false positives at a particular threshold. These concepts explain why the report includes both curves and confusion matrices and why a percentage shown in the interface is not called diagnostic certainty.

### 2.6.4 Explanation and trust
An attention map offers a way to inspect a computation, but it cannot compensate for weak predictive performance. A visually convincing overlay may invite excessive trust if it resembles a lesion. The interface therefore presents it as model attention and retains the clinical-review warning. Stronger explanation claims require separate experiments with appropriate reference annotations.

## 2.7 Synthesis
The literature supports building a study-level, multi-label research baseline with explicit dataset roles and several forms of evaluation. It does not support training indiscriminately on every available knee-related archive. The design chosen here preserves external datasets for reference evaluation, keeps calibration separate from fitting and uses a predefined internal criterion for model selection.

The resulting contribution is primarily integrative and empirical. The combination is useful because it reveals disagreements between internal selection and broader evaluation, not because every component is novel. A transparent negative finding can guide the next data-collection decision more effectively than a single optimistic accuracy value.

### 2.7.1 A hierarchy of claims
For this project, four levels of claim should be kept distinct. The first is computational: the software accepts a supported array and returns a finite score. The second is predictive: that score separates specified reference groups on a stated sample. The third is transferable: the behaviour persists in another population or acquisition setting. The fourth is clinical: using the system improves an intended decision or outcome. Evidence at one level does not automatically establish the next.

This hierarchy explains why the application can be complete enough for research testing while remaining a prototype. A real browser test addresses the first level. The MRNet metrics address the second under development limitations. The external comparisons probe part of the third, with weak results and imperfect references. Nothing in the recorded experiment directly measures the fourth. Reporting the strongest supported level is more informative than a single unqualified label such as accurate or complete.

### 2.7.2 Dataset quantity and fitness for purpose
A large file archive is not necessarily a large labelled training cohort. Multiple slices can belong to one study, multiple series can belong to one examination, and multiple examinations can belong to one patient. Counting files therefore cannot replace defining the independent unit. For the same reason, adding a second archive that contains reprocessed copies of an existing source may add storage without adding independent information.

The label question is equally important. A reconstruction dataset may provide high-quality images while lacking the references needed for supervised pathology classification. An ACL injury collection may be highly relevant to one output and uninformative about another. The present design interprets each resource by its supported task. This is a reasoned restriction of scope, not a failure to exploit every downloaded file.

### 2.7.3 What a fair comparison would require
An architecture comparison is easiest to interpret when the fitting data, evaluation data, preprocessing, optimisation budget and selection rule remain fixed. Where these differ, the observed score difference is a comparison of procedures rather than an isolated estimate of architecture effect. This distinction applies to the preserved baseline and current release because the calibration reservation changed the fitting allocation.

A subsequent controlled experiment could retain identical manifests and run each architecture under several fixed seeds. It could specify one selection criterion before training and retain a final evaluation cohort that is not consulted during iteration. This report recommends that design without pretending it was already completed. Its current model table answers which saved candidate won the implemented selection process, not which architecture is universally best.

## 2.8 Methodological approaches
The report uses CRISP-DM as an organising framework for understanding the implemented work. Chapman et al. (2000) describe a process covering business understanding, data understanding, preparation, modelling, evaluation and deployment. Here, deployment means a local research interface. This retrospective mapping does not claim that a preregistered CRISP-DM protocol governed every earlier development decision.

The primary experimental comparison is narrower than a benchmark competition. It compares ResNet-18 and EfficientNet-B0 under the project configuration and selects a checkpoint by internal AUROC. A later supplementary frozen-encoder benchmark also evaluated ResNet-50, DenseNet-121 and Swin Transformer under one locked split; it did not replace the primary selection or the active checkpoint. There is no multi-seed cross-validation, prospective trial or exhaustive hyperparameter search. These comparisons are informative only when their limitations remain attached to the reported results.

### 2.8.1 Distinguishing internal and external questions
Internal evaluation asks how the development procedure behaves within the available source and partitioning scheme. External evaluation asks how a fixed procedure behaves under a separately sourced set of observations and references. A dataset can be external by source yet no longer be untouched because its results have already been inspected. These are different dimensions of independence and should be reported separately.

The external cohorts in this project also change what can be scored. Three-target macro performance is meaningful only where all three references are available under suitable definitions. Reporting an ACL-only AUROC beside an MRNet macro AUROC does not make the quantities interchangeable. The report therefore keeps the target and unit attached to each number, rather than averaging incompatible external scores into a new headline metric.

### 2.8.2 Descriptive and causal explanations
The result that a candidate performs worse externally is descriptive. Explaining why requires more evidence. It might reflect different acquisition characteristics, sample composition, references, limited representation or some combination. Without controlled comparisons, selecting one of these explanations as the proven cause would exceed the experiment.

The same distinction applies to design choices. Weighted loss, augmentation and pooling are present in the implementation, but their separate benefits were not estimated. An ablation study would remove or change one component under matched conditions. Stating the implemented mechanism is appropriate; claiming that it caused a measured gain without such a comparison is not.

## 2.9 Tools
Python and PyTorch implement loading, preprocessing, neural modelling and inference. NumPy arrays represent the prepared stacks; configuration files centralise important parameters. Streamlit exposes the researcher workflow. Saved JSON and CSV results make metrics inspectable independently of screenshots, while notebooks support reproducible analysis and explanation of the pipeline.

Tool choice is justified by the required functions, not by a claim that a particular software stack guarantees scientific validity. The strongest practical safeguard is the connection between the displayed result and the exact saved checkpoint, configuration and dataset role. This makes a later experiment distinguishable from the release documented here.

### 2.9.1 Reproducibility as a chain of evidence
Reproducibility depends on preserving the relationship between input, transformation, model and result. A figure without its metric source can be difficult to audit. A checkpoint without its label order can be misinterpreted. A training script without the split manifest can reproduce a different experiment. For this reason, the local report links the narrative to run-specific artifacts and describes the responsibilities of each partition.

The evidence chain also has boundaries. A saved random seed does not guarantee identical output across every device or software release. A passed notebook may execute a demonstration using an existing checkpoint rather than repeat the complete training run. Accordingly, this report treats notebook execution as evidence that the recorded notebook path ran successfully, while training histories and saved model files support the training account.

### 2.9.2 User-facing presentation of uncertainty
The design problem is not solved by placing a disclaimer below an otherwise definitive diagnosis. The wording of the result, threshold status and attention caption must also match the evidence. A researcher should be able to distinguish an estimated score from a confirmed finding and identify the model that produced it. The present interface and report use decision-support language and retain the three-target scope.

This approach leaves a practical research question for future user testing: whether intended users actually understand the limitations. No usability study has answered that question here. A future evaluation could examine whether users can identify unsupported inputs, interpret thresholded statuses and avoid treating attention as lesion confirmation. Such an evaluation would produce new evidence rather than rely on the developer's assessment of a polished screen.

## 2.10 Comparison with existing work
Table 2.3 Relationship to direct knee MRI sources
| Work | Main relevance | Difference in this project |
| Bien et al. (2018) | Three-target knee MRI classification | Local encoder aggregation, partitions and calibration differ |
| Štajduhar et al. (2017) | ACL injury MRI research | Used as ACL-only external reference; previously evaluated |
| Zhao et al. (2022) | Pathology annotations for fastMRI | Local meniscal subset; incomplete negative references |
| KneeAssist AI release | Integrated local workflow and measured transfer | Research prototype; no new clinical validation |

Published scores are not placed in a league table against the local model because the relevant test conditions differ. A defensible direct comparison would require matched data, references, preprocessing and evaluation rules. The current comparison instead explains which ideas and resources are shared and where the experimental claims diverge.

The comparison also clarifies the project's intended contribution. A dataset paper can provide a resource, an architecture paper can provide a representation method, and a reporting guideline can improve transparency. KneeAssist AI combines these kinds of work into an inspectable application. The local evidence must establish whether that combination behaves adequately; citations cannot substitute for the missing experiment.

For a future decision-support study, the proposed use would need to be narrowed further. For example, assisting a researcher to inspect a known dataset and prioritising examinations in a clinical queue are different tasks. They involve different users, consequences and acceptable error tradeoffs. The present work evaluates the former type of research interaction. It has not measured queue prioritisation, reader assistance, patient outcomes or the consequences of delays caused by false negative predictions.

A useful comparison with subsequent studies should therefore record not only the encoder and AUROC but also the intended action, input completeness, reference standard and history of test-set access. This would make limitations comparable rather than hiding them in an overall score. Such a framework is a proposal arising from the present synthesis, not an additional completed experiment or a new validated assessment instrument.

## 2.11 Chapter summary
The review identified a defensible basis for the five objectives while separating methodological support from clinical evidence. Its main design implications are study-level processing, explicit label semantics, reserved calibration, transparent model selection and cautious interpretation of both external results and attention maps.

# CHAPTER 3 METHODOLOGY
## 3.1 Introduction
This chapter describes the implementation documented by the current project artifacts. It reports what was done rather than presenting an idealised protocol. The selected release is stored under runs/mri_finetune_02. Earlier artifacts remain relevant only where explicitly identified as the preserved baseline.

## 3.2 Research methodology
CRISP-DM is used to structure the work into six connected stages. Problem understanding defines the three findings and research-only use. Data understanding identifies which labels each source can support. Preparation converts supported studies into reproducible inputs. Modelling performs the primary two-encoder comparison and records the later five-family benchmark separately. Evaluation examines internal and external-reference behaviour. Deployment packages the selected model within a local interface (Chapman et al., 2000).

The mapping is descriptive rather than evidence of a registered prospective protocol. The distinction matters because previous exposure to official validation and KneeMRI results limits their status as untouched tests. These cohorts are still useful for descriptive evaluation, but the report does not erase the development history.

## 3.3 Research design
The study is a retrospective computational experiment using existing public research datasets and a local software verification component. It is not a clinical trial, reader study or investigation involving newly recruited patients. The principal comparison criterion is macro AUROC on the designated internal tuning set. The selected checkpoint is then reported on the official MRNet validation set and two external-reference subsets.

No external score was used to tune the selected release or reverse the predefined selection decision. This preserves the distinction between selecting a candidate and assessing its limitations. Nevertheless, because parts of the evaluation material had been used previously, an entirely independent estimate of future performance remains unavailable.

## 3.4 Data collection and preprocessing
### 3.4.1 Sources and access
Table 3.1 Dataset roles and local units
| Resource | Local quantity | Role and supported reference |
| MRNet | 1,250 studies | Sole training source; abnormality, ACL and meniscus labels |
| KneeMRI | 917 series grouped into 909 exams | External ACL injury reference only |
| fastMRI subset with fastMRI+ references | 199 files | Exploratory meniscal annotation reference |

The sources were obtained before this documentation task. This report does not redistribute archives or private access links. MRNet provides the training targets; KneeMRI and the fastMRI subset are retained separately. The original datasets remain preserved in the project storage. Source publications are Bien et al. (2018), Štajduhar et al. (2017) and Zhao et al. (2022), respectively; the local subset counts come from project manifests and evaluation artifacts.

### 3.4.2 Variables
The predictors are MRI intensity arrays organised by study and plane. The primary outcome is the general normal/abnormal label. The implementation also retains ACL and meniscus labels as secondary outputs. Study identifiers support grouping and traceability. Plane-availability indicators inform the model which stacks are supplied. No age, sex, symptoms or clinical history was incorporated into the reported model. Consequently, no demographic subgroup performance claim can be supported by this experiment.

### 3.4.3 Data quality
The audit checks study separation and exact series hashes. It also documents source-specific label meaning. Patient identifiers linking multiple examinations were not available, so the audit cannot certify patient-level independence. This is a material limitation: exact-duplicate detection is narrower than detection of related examinations, near-duplicates or repeated scans of the same person.

### 3.4.4 Cleaning and normalisation
For a supported volume, the pipeline uses within-volume intensity percentiles at the first and ninety-ninth percentiles, rescales the retained intensity range and resizes slices to 224 by 224 pixels. It applies the channel preparation and ImageNet normalisation expected by the pretrained encoder. Twelve evenly spaced slices per available plane provide a bounded computational input.

This standardisation makes numerical inputs consistent but does not harmonise scanner protocols or preserve every small structure. Sampling only twelve slices can omit a focal finding. Resizing can remove fine detail, and percentile scaling does not establish equivalence between acquisitions. These are modelling tradeoffs, not clinically validated preprocessing choices.

### 3.4.5 Integration
The three resources were not concatenated into a common training pool. Their labels, acquisition coverage and annotation completeness differ. MRNet alone supplies supervised fitting. External arrays are processed for compatible evaluation, but their distinct provenance and reference rules are retained. No additional disease label is inferred merely because a corresponding file is absent.

### 3.4.6 Feature construction
A pretrained two-dimensional encoder generates a feature vector for each sampled slice. Within each plane, mean and maximum pooling summarise the slice features. Plane summaries are concatenated with availability indicators and passed to a classification head. The resulting three logits are transformed into per-target scores for interpretation and thresholding.

### 3.4.7 Dimensionality reduction
Pooling reduces a variable or larger set of slice features to a fixed study representation. No principal component analysis or independently fitted feature-selection procedure is claimed. The architecture trades explicit spatial ordering for a compact summary; a future sequence-aware or three-dimensional approach could test whether this tradeoff contributes to missed pathology.

### 3.4.8 Sampling and partitions
Table 3.2 MRNet partition responsibilities
| Partition | Studies | Permitted use |
| Fitting | 804 | Fit model parameters and fitted feature transformations |
| Internal tuning | 226 | Select checkpoint and compare candidate architectures |
| Calibration | 100 | Fit calibration and select research thresholds |
| Official validation | 120 | Describe final performance; previously used in development |

The four partitions total 1,250 studies. The preserved baseline used 904 fitting studies before the separate calibration reservation. Comparisons with that baseline therefore reflect more than an isolated architecture change. Random seed 42 supports reproducibility, but only one seed was evaluated. Study members remain together rather than being split independently by slice.

The official validation subset contains 95 abnormality-positive, 54 ACL-positive and 52 meniscus-positive studies. These labels overlap because the targets are not mutually exclusive. Their sum must not be interpreted as a number of distinct patients. The evaluation reports each target separately and uses a macro average only after calculating the target-specific measures.

The internal tuning set determines selection, while the reserved calibration set determines the score mapping and threshold. These functions are kept separate even though both belong to development. The official validation set then describes performance of the selected procedure. Because it was evaluated earlier in the project, this final step remains development validation rather than a new independent test.

### 3.4.9 Privacy and governance
The interface accepts a case reference rather than requiring a name, and researcher use should employ pseudonymous references. This does not make the application a complete clinical information-governance system. De-identification, authorised data access, retention rules and institutional approval would need to be established before a prospective local study. No ethics approval number, patient consent process or hospital partnership is invented in this report.

### 3.4.10 Documentation
Dataset roles, metrics, calibration parameters and checkpoint identity are retained as machine-readable artifacts. This enables a reviewer to distinguish measured quantities from narrative interpretation. The report records rounded values for readability; JSON outputs retain the computational precision. The artifact index in Appendix C provides a route back to the experiment.

### 3.4.11 Versioning and storage
The release uses a run-specific directory so that current outputs can be distinguished from the earlier baseline. The selected checkpoint has a SHA-256 digest recorded in Appendix C and exported summaries. A digest identifies file contents; it does not validate the scientific adequacy of a model. Original datasets were preserved, and private links are excluded from this documentation.

### 3.4.12 RSNA online pilot and future review gate
The RSNA Knee Abnormality Detection competition data were inspected and used only in a private Kaggle pilot so that the large DICOM archive did not need to be copied to the local computer. The audit identified 4,407 unique studies, but only 58 studies had complete explicit labels for all 12 available conditions. The remaining report text was not converted into binary ground truth.

The pilot compared frozen ImageNet-pretrained ResNet-18 and EfficientNet-B0 encoders with grouped folds, per-series normalisation and study-level slice pooling. Exploratory macro AUROC was 0.6583 for ResNet-18 and 0.6583 for EfficientNet-B0. The small labelled cohort and reuse of folds for early stopping make these exploratory values unsuitable for a final model selection claim. The pilot was therefore kept separate from the deployed MRNet checkpoint.

Future RSNA work has a review gate rather than an automatic training step. The project contains a manifest builder, a two-reviewer agreement checker, a label template and an adjudication queue. No RSNA V2 model has been trained because qualified independent labels are not available. This is a deliberate restriction: a script can check agreement, but it cannot create a trustworthy clinical reference standard.

## 3.5 Model selection and architecture
Both ResNet-18 and EfficientNet-B0 use ImageNet-pretrained slice encoders. The study head has a 128-unit hidden layer, ReLU activation, dropout of 0.5 and three output logits. This is a multi-label architecture: a study can receive more than one positive finding. The code separates data loading, modelling, evaluation and inference so that an additional verified target could be introduced through an explicitly retrained model.

The comparison includes a frozen-encoder warm-up and attempted partial fine-tuning. The winning release is the EfficientNet-B0 frozen warm-up checkpoint followed by calibration. Describing it as a successful improved fine-tuned encoder would be inaccurate. The selection criterion was the internal macro AUROC, not training accuracy or external performance.

## 3.6 Training and evaluation procedure
Weighted binary cross-entropy with logits addresses label imbalance by using negative-to-positive ratios from the fitting labels. AdamW performs optimisation. Learning-rate scheduling, early stopping and checkpoint saving support controlled training. Training and validation histories are retained. Resume support restores saved state rather than starting an indistinguishable duplicate experiment.

The warm-up uses a head learning rate of 0.0003, a maximum of 40 epochs and early-stopping patience of three. Partial fine-tuning uses encoder and head learning rates of 0.00001 and 0.00003, respectively, with a maximum of six epochs and patience of three. Physical study batch size is one with accumulation over four studies during fine-tuning; warm-up batches use cached features. Batch-normalisation layers and earlier encoder layers remain frozen.

The device used is an NVIDIA T500 with 4 GB GPU memory. Full half-precision encoder execution produced non-finite behaviour, so the encoder uses full precision and mixed precision is restricted to the head where supported. Gradient scaling and non-finite-step handling are retained. This is a documented numerical compromise; the report does not claim that every component trained successfully in half precision.

## 3.7 Optimisation and calibration
Training augmentation includes small rotations up to approximately five degrees, scale variation from 0.95 to 1.05 and modality dropout with probability 0.1 during the fine-tuning procedure. No horizontal-flip policy, exhaustive grid search or Bayesian optimisation is claimed. These settings form the implemented experiment, not a proven optimum.

After model selection, positive-slope Platt calibration is fitted using the reserved 100 studies. Per-target research thresholds maximise Youden's index on that partition. Holding the slope positive preserves score ordering. The calibration sample is small, and no external calibration guarantee follows. Thresholds were not adjusted after seeing the external results.

## 3.8 Validation and software testing
The final model is assessed on 120 official validation studies. KneeMRI series are grouped into 909 exam identifiers, with repeated-series scores averaged and the maximum recorded injury severity used for the exam reference. The fastMRI subset contains 199 files with meniscal reference status derived from matching annotations. These procedures are documented separately to avoid treating the cohorts as equivalent.

Software verification covers automated tests, executed notebooks and a real browser workflow. The final release records 33 passing automated tests and nine executed notebooks. A real MRI case was uploaded and used to verify predictions, attention, export controls and clearing. These checks establish that the tested paths work; they do not establish clinical accuracy, comprehensive security or usability for all intended users.

Input checks require finite numeric three-dimensional arrays and reject unsupported object arrays, invalid dimensions and constant-intensity sequences. Plane names identify axial, coronal or sagittal input. Duplicate stacks for the same plane are rejected, and ZIP contents are read without extracting arbitrary paths to disk.

The updated interface also accepts a bounded DICOM ZIP. It requires `.dcm` files arranged in explicitly named axial, coronal and sagittal folders, accepts no more than one series for a plane, reads the archive in memory and checks DICOM orientation when it is present. Inconsistent dimensions, unreadable compressed slices, unsafe archive paths and geometry that conflicts with the stated plane are rejected. This feature makes supported research inputs easier to load; it does not verify de-identification, anatomy, scanner suitability, series choice, patient identity or clinical applicability.

The workflow test therefore answers a bounded question: can a supported real example complete the intended path with traceable output? It does not estimate the rate at which unsuitable clinical inputs will be detected. That rate would require a deliberately assembled set of valid and invalid examples with an evaluation protocol. The distinction is retained so that functional testing is not mistaken for input-distribution validation.

## 3.9 Statistical analysis
For each target, the evaluation reports AUROC, F1, sensitivity, specificity, precision, accuracy and a confusion matrix. Macro values are arithmetic means across the three targets, not a patient-level all-findings-correct rate. ROC and precision–recall curves show threshold tradeoffs. Brier scores describe squared probability error on the stated cohort.

Sensitivity is TP divided by TP plus FN; specificity is TN divided by TN plus FP; precision is TP divided by TP plus FP; accuracy is TP plus TN divided by all observations. F1 is twice precision times sensitivity divided by their sum. These definitions make clear which denominator changes when the cohort prevalence or threshold changes.

AUROC uncertainty uses 1,000 study-bootstrap resamples. No formal paired superiority test or multiple-comparison procedure was performed. The confidence intervals are conditional on the selected model and evaluation sample; they do not incorporate training-seed variability, selection bias or unknown within-patient dependence. Differences between candidates are therefore reported descriptively rather than as statistically significant improvements.

## 3.10 Tools and reproducibility
The implementation uses Python, PyTorch, NumPy, configuration files, saved checkpoints, metric outputs and Streamlit. Important settings are centralised rather than scattered through the interface. The inference pipeline loads the saved model independently of a training notebook. A reproducibility review should inspect the run configuration, source version, dependencies, split manifests and checkpoint together; the checkpoint alone is insufficient.

## 3.11 Ethical considerations
The prototype does not replace radiologists, establish a definitive diagnosis or recommend treatment. It predicts only three findings, and a low score cannot exclude other pathology. Grad-CAM is explicitly described as model attention. The report includes an AI-use declaration because generative AI assisted the project and this documentation; it does not present generated prose or code as unaided work.

## 3.12 Chapter summary
The method combines defined dataset roles, a primary two-encoder comparison, a separately retained five-family benchmark, study aggregation, reserved calibration and separate evaluation. Its main limitations are incomplete patient linkage, reused evaluation material, restricted external references and a single-seed experiment. These limits remain part of the interpretation of every result in the next chapter.

# CHAPTER 4 RESULTS AND DISCUSSION
## 4.1 Introduction
Results are organised by the five objectives and drawn from the preserved run artifacts. Figures reproduce project-generated outputs rather than illustrative performance. Numerical values are rounded for presentation. The selected model is the calibrated EfficientNet-B0 frozen warm-up checkpoint, and the main evaluation set is the 120-study MRNet official validation partition.

## 4.2 Findings by objective
### 4.2.1 Objective 1 Data preparation and auditing
The project prepared 1,250 MRNet studies and separated them into 804 fitting, 226 tuning, 100 calibration and 120 official validation studies. The general abnormality label is used as the primary normal/abnormal task. KneeMRI and the fastMRI subset remained outside supervised fitting. Study-level and exact-series checks were implemented. The objective is achieved at study level, with the explicit qualification that patient-level independence could not be established.

The external resources were useful for different questions. KneeMRI supplied an ACL reference for 909 exam groups; the fastMRI subset supplied 199 meniscal annotation comparisons. They did not fill the gap for a complete, independently confirmed three-target external evaluation. No extra dataset was claimed to make the prototype clinically complete.

### 4.2.2 Objective 2 Model development and comparison
Table 4.1 Internal selection results
| Candidate | Internal macro AUROC | Interpretation |
| Preserved incumbent | 0.8478 | Earlier baseline with a different fitting allocation |
| ResNet-18 fine-tuned candidate | 0.8485 | Best reported fine-tuned ResNet candidate |
| EfficientNet-B0 frozen warm-up | 0.8532 | Selected by internal AUROC |

The selected candidate exceeded the incumbent's internal AUROC by approximately 0.0054. This small descriptive difference is not a demonstrated improvement in generalisation. Partial fine-tuning did not displace the frozen EfficientNet-B0 checkpoint. The objective of developing and comparing both architectures was achieved, but an expectation that fine-tuning would necessarily perform better was not supported.

After the primary selection, a supplementary frozen-encoder benchmark compared five MRNet-only candidates on one locked internal-tuning split: ResNet-18 (0.8393), ResNet-50 (0.8275), DenseNet-121 (0.8354), EfficientNet-B0 (0.8472) and Swin Transformer (0.8395) macro AUROC. EfficientNet-B0 was the best new candidate, but it did not exceed the active calibrated baseline (0.8478 on the same internal criterion). The new candidate was therefore not promoted. This is a development comparison, not evidence that one architecture is clinically superior: the active model has a different historical fitting allocation, the benchmark used one seed, and no external cohort was used for selection.

### 4.2.3 Objective 3 Evaluation and calibration
Table 4.2 MRNet official validation performance
| Finding | AUROC | F1 | Sensitivity | Specificity | Precision | Accuracy |
| General abnormality | 0.901 | 0.927 | 0.937 | 0.680 | 0.918 | 0.883 |
| ACL tear | 0.825 | 0.746 | 0.815 | 0.697 | 0.688 | 0.750 |
| Meniscal tear | 0.754 | 0.631 | 0.673 | 0.647 | 0.593 | 0.658 |
| Macro average | 0.827 | 0.768 | 0.808 | 0.675 | 0.733 | 0.764 |

General abnormality is the primary result for the narrowed academic topic and had the strongest observed ranking and threshold performance. Meniscal tear was weaker, with precision of 0.593 and sensitivity of 0.673; it remains an additional research output, not the main normal/abnormal claim. The macro accuracy of 0.764 must not be advertised as the probability that every finding for a new patient is correct. It averages three binary accuracies on this particular cohort.

Calibration and threshold selection were completed on the reserved subset. The thresholds are approximately 0.6944 for general abnormality, 0.2461 for ACL tear and 0.4808 for meniscal tear. Different thresholds explain why a single common 50% rule would not reproduce the reported statuses. These thresholds are research operating points, not clinically approved decision rules.

### 4.2.4 Objective 4 Independent inference and attention
Independent inference loads the selected checkpoint and processes a supported MRI study without running a training notebook. Saved results include per-target scores and checkpoint provenance. Grad-CAM generation was verified on a real case. The objective is achieved as a functioning research inference path, but lesion localisation accuracy and explanation faithfulness have not been validated.

An example case used for workflow testing returned approximately 58.0% general abnormality, 8.6% ACL tear and 2.8% meniscal tear. These values illustrate the application's output, not a demonstration of diagnosis. A single example cannot estimate performance, and the general-abnormality value is below the selected general-abnormality threshold despite exceeding 50%.

### 4.2.5 Objective 5 Local interface verification
The Streamlit workflow supports a case reference, MRI array, NIfTI, DICOM or ZIP upload, intake inspection, manual plane confirmation where metadata is absent, analysis, slice viewing, prediction display, attention, export and clearing. The supported ZIP route includes arrays, NIfTI volumes and guarded DICOM series. JPEG/PNG viewing and prediction are excluded. Browser verification confirmed the real-case upload and analysis path, attention display, exported JSON and clearing behaviour. Readable empty-input handling and dedicated NPY, NIfTI and DICOM input-safety tests were also verified.

The final release evidence records 33 automated tests passing and nine notebooks executed successfully. The dashboard also records a guarded ensemble and localisation framework, which remains disabled because there is no held-out evidence of an improved ensemble and no verified lesion-location labels for detection or segmentation. These are functional results for the tested version. They do not establish that every malformed archive, unsupported scan or clinical workflow has been covered. The interface is ready for controlled research testing with supported inputs, not unrestricted patient care.

## 4.3 Descriptive results and visualisations
Table 4.3 Label distribution in the 120-study MRNet evaluation cohort
| Finding | Positive reference | Negative reference | Positive proportion |
| General abnormality | 95 | 25 | 79.2% |
| ACL tear | 54 | 66 | 45.0% |
| Meniscal tear | 52 | 68 | 43.3% |

The high proportion of abnormal studies makes the general-abnormality precision and accuracy dependent on a case mix that may differ from future use. The small group of 25 negative abnormality references also makes specificity sensitive to a few cases. These counts are a reason to retain the confusion matrix beside summary metrics.

@figure runs/mri_finetune_02/results/resnet18/training_validation_curves.png | Figure 4.1 Recorded ResNet-18 fine-tuning history. Source: project training artifacts. The curve describes this run and does not establish multi-seed stability.

@figure runs/mri_finetune_02/results/efficientnet_b0/training_validation_curves.png | Figure 4.2 Recorded EfficientNet-B0 fine-tuning history. Source: project training artifacts. Fine-tuning did not exceed the selected frozen warm-up checkpoint.

The histories document the attempted optimisation stages rather than a guaranteed monotonic improvement. ResNet-18 ran six fine-tuning epochs, while EfficientNet-B0 ran three before stopping. The selected EfficientNet checkpoint corresponds to the frozen warm-up stage, recorded as selection epoch zero in the fine-tuning comparison. This should not be interpreted as an untrained random model: the pretrained encoder and fitted warm-up head preceded that stage.

@figure runs/mri_finetune_02/results/final/roc_curves.png | Figure 4.3 ROC curves on the 120 MRNet official validation studies. Source: project evaluation artifacts, 27 September 2026.

The ROC curves describe ranking over possible thresholds. They do not select a clinical threshold or quantify the consequences of an error. The ordering of AUROCs is consistent with the tabulated results, with general abnormality stronger than meniscal tear on this cohort.

@figure runs/mri_finetune_02/results/final/precision_recall_curves.png | Figure 4.4 Precision–recall curves for the same MRNet cohort. Source: project evaluation artifacts.

Precision–recall behaviour provides another view of positive-finding performance. Comparisons across targets should account for their different positive proportions. The curves support inspection of tradeoffs, while the table reports the specific calibrated-threshold operating points used by the release.

## 4.4 Confusion matrices and external performance
@figure runs/mri_finetune_02/results/final/confusion_matrices.png | Figure 4.5 Confusion matrices at the selected research thresholds. Source: project evaluation artifacts. Rows represent reference labels and columns predictions.

For general abnormality, 89 positive studies were flagged and six were missed; eight of 25 negative-reference studies were flagged. For ACL tear, 44 positives were flagged and ten were missed, with twenty false positives. For meniscal tear, 35 positives were flagged and seventeen were missed, with twenty-four false positives. These counts show why apparently strong aggregate performance cannot be interpreted as error-free assistance.

Table 4.4 External-reference results
| Evaluation | Units | AUROC | Sensitivity | Specificity | Precision | F1 |
| KneeMRI ACL | 909 exams | 0.603 | 0.549 | 0.586 | 0.305 | 0.392 |
| fastMRI meniscal reference | 199 files | 0.656 | 0.709 | 0.552 | 0.629 | 0.667 |

The KneeMRI ACL comparison included 226 injury-reference and 683 healthy-ACL-reference exams. It produced 124 true positives, 102 false negatives, 400 true negatives and 283 false positives. These errors are substantial. The healthy-ACL category should not be interpreted as absence of all knee abnormalities.

The fastMRI comparison included 103 files with a positive meniscal annotation reference and 96 without that reference. The resulting counts were 73 true positives, 30 false negatives, 53 true negatives and 43 false positives under the implemented reference convention. Because annotations are non-exhaustive, these labels do not provide a definitive clinical sensitivity or specificity estimate.

@figure runs/mri_finetune_02/results/external_kneemri/roc_curves.png | Figure 4.6 KneeMRI ACL external-reference ROC curve. Source: project evaluation artifacts. This previously evaluated cohort is not a new untouched test.

@figure runs/mri_finetune_02/results/external_fastmri/roc_curves.png | Figure 4.7 Exploratory fastMRI meniscal-reference ROC curve. Source: project evaluation artifacts. Annotation absence is not a definitive normal reference.

## 4.5 Inferential statistics
Table 4.5 Study-bootstrap AUROC intervals
| Finding | Point estimate | 95% bootstrap interval |
| General abnormality | 0.901 | 0.820–0.966 |
| ACL tear | 0.825 | 0.750–0.892 |
| Meniscal tear | 0.754 | 0.666–0.836 |
| Macro average | 0.827 | 0.769–0.876 |

These intervals are based on 1,000 resamples of the evaluation studies. They quantify only the uncertainty represented by that resampling scheme. They are not confidence intervals for performance at a new Zimbabwean hospital, and they do not justify a statistical superiority claim over the earlier baseline.

## 4.6 Statistical analysis of the release comparison
Table 4.6 Earlier baseline and selected release
| Evaluation AUROC | Earlier baseline | Selected release |
| MRNet macro | 0.8499 | 0.8267 |
| KneeMRI ACL | 0.6346 | 0.6030 |
| fastMRI meniscal reference | 0.6740 | 0.6561 |

All three reported broader evaluation AUROCs decreased, despite the higher internal selection AUROC. This is the central negative finding. Because the fitting allocation and calibration procedure also changed, the comparison does not isolate the effect of architecture or fine-tuning. It does establish that the selected release should not be described as an overall performance improvement.

General-abnormality specificity increased from 20% in the earlier release to 68%, while sensitivity decreased from 100% to 93.7%. This is an operating-point tradeoff rather than proof of greater clinical value. Final Brier scores were approximately 0.098 for abnormality, 0.192 for ACL and 0.200 for meniscus; without a prespecified comparative calibration analysis, these values alone do not establish successful transportable calibration.

## 4.7 Attention visualisation
@figure runs/mri_finetune_02/results/final/example_1130.png | Figure 4.8 Saved MRI example and attention visualisation. Source: project inference artifact. The overlay is model attention, not a confirmed lesion.

The displayed example demonstrates that the explanation pipeline produces an overlay associated with a prediction. It does not establish that highlighted pixels identify the anatomical cause of a tear. No expert-drawn lesion masks were available for a localisation score. The appropriate conclusion is functional explanation output with unresolved explanatory validity.

## 4.8 Anomalies and numerical behaviour
The training work encountered non-finite behaviour during half-precision encoder computation. The final numerical configuration retained full-precision encoder execution, with mixed precision used only where stable. The issue was treated as a computational failure to diagnose rather than evidence that a clinical case should be excluded.

No claim is made that unusual clinical studies were systematically identified and removed. The system lacks a validated out-of-distribution detector and does not automatically confirm knee anatomy or patient consistency. Unsupported content can therefore remain a scientific risk even when a file passes basic structural checks.

## 4.9 Robustness
Only one training seed was used, and no repeated cross-validation result is available. Missing-plane support in the architecture does not establish that each missing-plane combination is accurate. The external cohorts differ in plane coverage and reference construction, but the experiment does not isolate those factors through controlled ablations.

Similarly, successful processing on the local GPU does not establish equivalent performance and latency on every machine. The saved checkpoint and inference tests support reproducibility of this release, while a broader robustness programme would require repeated seeds, additional devices, acquisition strata and independently labelled cases.

## 4.10 Discussion
### 4.10.1 Interpretation of the main findings
The five engineering and research objectives were addressed, but their completion has different meanings. Data preparation, model comparison, inference and interface verification are observable technical achievements. Evaluation is also an achievement when it reveals weak generalisation. It is not necessary to redefine success as high accuracy to justify reporting the work.

The internal selection rule chose a candidate that did not improve broader evaluation. This outcome suggests that the designated tuning result is an incomplete guide to transfer performance in the present experiment. It does not establish a single cause. Differences in acquisition, sample composition, limited training size, label semantics and ordinary sampling variation are plausible explanations requiring further controlled study.

### 4.10.2 Relationship to the literature
The use of a study-level three-target task follows the broad MRNet research setting, but the architecture and partitions differ from the published study (Bien et al., 2018). The present results cannot be used as a direct replication claim. More generally, the discrepancy between internal and external performance is consistent with the importance of evaluating transfer under changed data conditions.

Reporting guidance helps make these differences visible, but it cannot correct them retrospectively. Similarly, attention output offers an inspection mechanism without demonstrating a clinically faithful explanation. The literature therefore supports restraint in interpretation and a clearly specified next evaluation, rather than a stronger marketing claim.

### 4.10.3 Limitations
The principal limitations are unavailable patient linkage, reused official validation and KneeMRI material, limited fastMRI references, one seed, no local clinical cohort and no prospective user study. The guarded DICOM route is narrower than routine clinical imaging workflows because it requires explicit plane folders and does not select sequences automatically. The three targets do not cover the full diagnostic content of a knee MRI examination. No weighted ensemble is deployed because its benefit has not been demonstrated on held-out calibrated predictions; detection boxes and segmentation masks are unavailable because suitable verified lesion-location annotations are absent.

These limitations affect different claims. Missing patient linkage constrains independence; non-exhaustive annotations constrain reference validity; one seed constrains training robustness; and absent clinical workflow evidence constrains usefulness. They cannot all be resolved by adding another architecture or making the dashboard more polished.

## 4.11 Chapter summary
The release works as a research application and produces auditable predictions and attention. Its strongest measured performance is general abnormality on MRNet. External-reference results are weak, and the selected release did not improve the broader AUROCs over the earlier baseline. The findings support controlled research testing and a better independent evaluation design.

# CHAPTER 5 CONCLUSIONS AND RECOMMENDATIONS
## 5.1 Introduction
This chapter draws conclusions at the level supported by the experiment. It separates achieved functions from unresolved scientific and clinical questions. The project is complete as a documented research prototype, while clinical suitability remains unestablished.

## 5.2 Summary by objective
Table 5.1 Achievement and remaining qualification
| Objective | Achieved outcome | Qualification |
| 1 | Prepared data and study-level audits | Patient-level independence not proven |
| 2 | Compared both requested encoders and retained a five-family benchmark | Frozen EfficientNet selected; no new candidate exceeded the active baseline |
| 3 | Calibrated and reported internal and external-reference results | Reused cohorts and limited references constrain conclusions |
| 4 | Independent inference and Grad-CAM output | Attention is not validated lesion localisation |
| 5 | Verified local MRI workflow and exports | Research usability only; no clinical field study |

The objectives were deliberately function-based and measurable. They did not promise a numerical accuracy level, replacement of specialists or deployment certification. This allows the report to describe completed work without suppressing the observed weaknesses.

## 5.3 Conclusions
The project demonstrates that an end-to-end knee MRI research application can be assembled around pretrained image encoders, study-level pooling and a modular inference pipeline. It also demonstrates that a higher internal selection score does not necessarily translate to higher performance on other evaluation material. The chosen EfficientNet-B0 frozen warm-up release achieved MRNet macro AUROC of 0.8267, while external ACL and meniscal-reference AUROCs were 0.6030 and 0.6561.

The appropriate conclusion is research readiness with important validity limits. The application can be used to inspect supported studies, reproduce outputs and guide subsequent experiments. It should not be used to rule out injury, produce autonomous diagnoses or imply comprehensive assessment of knee pathology.

## 5.4 Implications
### 5.4.1 Theoretical implications
The results reinforce the distinction between model selection and generalisation in this particular implementation. They do not establish a new general theory about EfficientNet or ResNet. The negative result is useful because it identifies a gap between the optimisation target and the broader evaluation outcomes that future work can investigate.

### 5.4.2 Practical implications
A usable application makes research inspection easier, but the interface must preserve uncertainty rather than hide it behind a percentage. Checkpoint provenance, clear input requirements and exportable summaries help researchers understand what produced a result. Before clinical integration, the workflow would require independent accuracy evidence, appropriate governance and evaluation with intended users.

### 5.4.3 Education 5.0 relevance
The project provides a teaching and research artifact combining data handling, machine learning and application development. Its innovation is an implemented local workflow with documented limits. Industrialisation, commercial viability, improved service delivery and community health benefit have not been measured; they remain possible future directions rather than achieved Education 5.0 outcomes.

## 5.5 Limitations of the study
The evaluation sample is modest, and only one training seed was assessed. Patient-level linkage is missing, while important evaluation resources have already informed earlier development. The fastMRI comparison uses incomplete annotation references. There are no Zimbabwean patients, no prospective outcomes and no independent radiologist reader study.

The input pipeline supports prepared MRI arrays and a bounded DICOM ZIP route rather than the full range of clinical formats. Slice sampling and resizing may discard useful information. Availability masks and augmentation do not substitute for explicit testing of missing planes or acquisition changes. No evidence establishes reliable identification of out-of-distribution inputs.

## 5.6 Recommendations
The first priority is an independently governed evaluation cohort with complete references for all three targets, adequate positive and negative cases, and patient identifiers suitable for grouping. The protocol should be fixed before results are viewed. Acquisition characteristics and relevant subgroups should be retained where lawful so that transfer and uncertainty can be examined rather than guessed.

The second priority is a controlled experimental comparison. Repeated seeds, matched partitions and ablations of slice sampling, pooling and plane coverage would help identify which choices matter. Any retraining should preserve an untouched final test cohort. External failures should guide a new research protocol, not repeated threshold tuning against the same evaluation cases.

The third priority is input and workflow validation. The current DICOM route demonstrates guarded import, but routine clinical-format support would still require controlled series selection, orientation verification and metadata governance. Anatomical checks, mixed-case prevention and out-of-distribution assessment should be evaluated explicitly. Intended users should test the interface in a research setting before claims about usability or efficiency are made.

These priorities should be implemented in order of the uncertainty they resolve. A new dashboard feature cannot determine whether a negative-reference label is correct. A larger encoder cannot establish that two studies belong to different patients. Completing the data and evaluation protocol first would make subsequent model changes easier to interpret and reduce the chance of repeatedly optimising to the same development material.

Acceptance criteria for any next release should be written before its final evaluation. They should specify the target population, unit of analysis, completeness of reference labels, required uncertainty reporting and the consequences of each type of error. This report does not invent numerical clinical acceptance thresholds, because no intended clinical decision or stakeholder-approved error-cost framework has been established. A qualified clinical collaborator and the relevant institution would need to participate in defining such criteria.

## 5.7 Future work
Future modelling could compare sequence-aware aggregation or three-dimensional representations with the current pooled-slice baseline. Such work should use controlled compute budgets and identical evaluation rules. Additional targets should be introduced only when their labels and clinical meanings are sufficiently supported, followed by new evaluation rather than an unchanged claim of model completeness.

Explanation research should include sanity checks and, where available, expert localisation references. Calibration should be examined with enough cases and on independent data, with operating thresholds selected for a specified use and error-cost profile. A local prospective study would be a separate undertaking requiring institutional arrangements; this report does not imply that these have already been secured.

## 5.8 Final conclusion
KneeAssist AI fulfils the five documented objectives as a functioning and evaluated research prototype. Its most valuable output is a transparent system whose strengths and failures can both be inspected. The next step is independent, well-referenced validation and controlled improvement, not a claim that the model is ready to replace clinical judgement.
