"""Create a printable, evidence-bound defence brief for KneeAssist AI."""

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(r"C:\Users\HP\Desktop\KneeAssist-AI")
OUT = ROOT / "docs" / "academic" / "submission" / "KneeAssist_AI_Defence_Cheat_Sheet.docx"

BLUE = "12344B"
PALE = "EAF2F8"
GREY = "D9E2F3"


def shade(cell, fill: str) -> None:
    properties = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    properties.append(shading)


def set_cell_text(cell, text: str, bold: bool = False) -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(2)
    run = paragraph.add_run(text)
    run.bold = bold
    run.font.size = Pt(9)
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER


def field(paragraph, instruction: str) -> None:
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    separate.append(text)
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.append(begin)
    run._r.append(instr)
    run._r.append(separate)
    run._r.append(end)


def add_heading(doc: Document, text: str, level: int = 1) -> None:
    p = doc.add_paragraph(style=f"Heading {level}")
    p.add_run(text)


def add_bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(3)
        p.add_run(item)


def add_table(doc: Document, headers: list[str], rows: list[list[str]], widths: list[float] | None = None) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.autofit = False
    for index, header in enumerate(headers):
        cell = table.rows[0].cells[index]
        shade(cell, BLUE)
        set_cell_text(cell, header, bold=True)
        for run in cell.paragraphs[0].runs:
            run.font.color.rgb = RGBColor(255, 255, 255)
    for row_index, row in enumerate(rows):
        cells = table.add_row().cells
        for column, value in enumerate(row):
            if row_index % 2:
                shade(cells[column], PALE)
            set_cell_text(cells[column], value)
    if widths:
        for row in table.rows:
            for index, width in enumerate(widths):
                row.cells[index].width = Inches(width)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_qa(doc: Document, question: str, answer: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run("Question: " + question)
    r.bold = True
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.2)
    p.paragraph_format.space_after = Pt(5)
    p.add_run("Answer: " + answer)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.65)
    section.bottom_margin = Inches(0.65)
    section.left_margin = Inches(0.75)
    section.right_margin = Inches(0.75)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(10)
    normal.paragraph_format.space_after = Pt(5)
    for level, size in ((1, 15), (2, 12), (3, 11)):
        style = styles[f"Heading {level}"]
        style.font.name = "Aptos Display"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(BLUE)
        style.paragraph_format.space_before = Pt(10 if level == 1 else 7)
        style.paragraph_format.space_after = Pt(4)
    if "Defence subtitle" not in styles:
        style = styles.add_style("Defence subtitle", WD_STYLE_TYPE.PARAGRAPH)
        style.font.name = "Aptos"
        style.font.size = Pt(11)
        style.font.color.rgb = RGBColor.from_string(BLUE)

    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    header.add_run("KneeAssist AI | Defence brief | Research prototype").font.size = Pt(8)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run("Page ").font.size = Pt(8)
    field(footer, "PAGE")

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("KneeAssist AI Defence Cheat Sheet")
    run.bold = True
    run.font.name = "Aptos Display"
    run.font.size = Pt(22)
    run.font.color.rgb = RGBColor.from_string(BLUE)
    subtitle = doc.add_paragraph(style="Defence subtitle")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.add_run("Study-level classification of normal and abnormal knee MRI studies")
    intro = doc.add_paragraph()
    intro.alignment = WD_ALIGN_PARAGRAPH.CENTER
    intro.add_run("Use this to explain the implemented system accurately. Do not claim features, clinical validation or personal work that you cannot demonstrate.").italic = True

    add_heading(doc, "1. Your 30-second opening", 1)
    doc.add_paragraph(
        "My project is KneeAssist AI, a local research prototype that processes a knee MRI study and produces study-level scores for general abnormality, ACL tear and meniscal tear. "
        "I trained the active EfficientNet-B0 model on MRNet data only. The main academic task is normal-versus-abnormal MRI study classification; ACL and meniscal scores are additional research outputs. "
        "The dashboard accepts supported MRI files, checks the input structure, shows the scores and displays Grad-CAM as a coarse explanation of what influenced a score. It is not a diagnostic, detection or segmentation system."
    )
    add_heading(doc, "2. The problem, aim and scope", 1)
    add_table(doc, ["Item", "Defensible explanation"], [
        ["Problem", "A knee MRI study contains many slices and may take time to review. The project investigates whether deep learning can classify a complete study as normal or abnormal using public research data."],
        ["Aim", "To develop and evaluate a reproducible deep-learning system for study-level classification of normal and abnormal knee MRI studies, with transparent evidence and a local interface."],
        ["Primary scope", "Study-level general abnormality classification from MRI. It combines sampled slices rather than predicting from one internet image or photograph."],
        ["Additional outputs", "ACL tear and meniscal tear research scores. They are weaker than the main general-abnormality result and must not be presented as diagnoses."],
        ["Out of scope", "Clinical deployment, radiologist replacement, grading tear severity, lesion boxes, segmentation masks, X-rays, photographs, and unvalidated screenshots."],
    ], [1.2, 5.9])

    add_heading(doc, "3. Explain the whole system in one flow", 1)
    add_table(doc, ["Stage", "What happens", "Why it is needed"], [
        ["1. Upload", "The user uploads one supported knee MRI study: NPY stack, NIfTI volume, DICOM series or supported ZIP.", "The model needs a 3D MRI volume, not a single screenshot."],
        ["2. Intake checks", "The app checks file type, readable content, dimensions, slice count and available plane metadata.", "This prevents common malformed-input errors. It cannot prove anatomy, patient identity or clinical suitability."],
        ["3. Plane confirmation", "Axial, coronal and sagittal orientation is read from reliable metadata where available. Metadata-free arrays require manual confirmation.", "Plane affects how the model interprets image structure. Filenames alone are not trusted."],
        ["4. Preprocessing", "Each sequence is normalised using the 1st and 99th intensity percentiles, resized to 224 × 224 and sampled to 12 slices per plane.", "This makes input dimensions and intensity scale consistent with training."],
        ["5. Encoder", "ImageNet-pretrained encoders convert sampled MRI slices into image features. The current dashboard routes DenseNet-121 to general abnormality, ResNet-18 to ACL and Swin-T to meniscus.", "Transfer learning gives the model useful image representations before MRI-specific fitting."],
        ["6. Study aggregation", "Mean and maximum pooling combine slice features within each plane; plane masks show which sequences were supplied.", "The prediction represents the study, not an individual slice."],
        ["7. Output", "A multi-label head produces three independent scores: general abnormality, ACL tear and meniscal tear. Calibrated research thresholds determine the research-review flag.", "Several findings can be positive at the same time; the threshold is not a clinical rule."],
        ["8. Explanation", "Grad-CAM overlays positive contribution to a selected score on one selected sequence and sampled slice.", "It supports inspection of model behaviour, but is not lesion localisation."],
    ], [1.25, 3.35, 2.5])

    add_heading(doc, "4. Data, split and training", 1)
    add_table(doc, ["Component", "What to say"], [
        ["Training source", "MRNet-v1.0 was the only supervised training dataset. It contains knee MRI studies with labels for abnormality, ACL tear and meniscal tear."],
        ["Split", "The preserved protocol used 804 fitting studies, 226 internal-tuning studies, 100 reserved calibration studies and 120 official-validation studies."],
        ["Leakage control", "The project uses study-level separation and exact-series checks. Patient linkage was unavailable, so I cannot claim patient-level independence."],
        ["Class imbalance", "Weighted BCEWithLogitsLoss gives more importance to minority positive labels during fitting."],
        ["Training choices", "Fixed random seed, ImageNet-pretrained encoders, AdamW, learning-rate scheduling, early stopping, checkpoint saving and CUDA mixed precision where available."],
        ["Model selection", "ResNet-18 and EfficientNet-B0 were compared using internal-tuning macro AUROC; EfficientNet-B0 is the formal historical baseline. The later portfolio supplies fixed live routes: DenseNet-121 for general abnormality, ResNet-18 for ACL and Swin-T for meniscus."],
    ], [1.45, 5.65])

    add_heading(doc, "5. Dashboard walkthrough: explain every visible part", 1)
    add_table(doc, ["Dashboard area", "What it does", "How to explain it in defence"], [
        ["Research warning", "Appears at the top of the app.", "It sets the intended use: this is research decision support, not autonomous diagnosis."],
        ["Model evidence and supported inputs", "Shows active target-routing evidence, formal baseline provenance and supported file types.", "It makes provenance visible and states that technical file checks are not clinical validation."],
        ["Add a study", "Accepts NPY, NIfTI, DICOM and ZIP study files plus a case reference.", "JPEG/PNG, X-rays and internet screenshots are rejected because the model was not trained or validated on them."],
        ["Study intake", "Lists type, shape, detected plane, confidence and status; asks for manual plane confirmation if needed.", "The app avoids silently guessing orientation when metadata is missing."],
        ["Analyse study", "Loads the three compatible registered target models, preprocesses the MRI and calculates study-level scores.", "The registry has three fixed validated research routes. Each finding is assigned before scoring; this is target routing rather than a weighted ensemble or patient-specific winner selection."],
        ["Research scores", "Shows each probability, research threshold and research-review status.", "A probability is a model output, not the chance that a patient has a confirmed diagnosis."],
        ["Distance from threshold", "Shows whether a score is near or far from a research threshold.", "It is not confidence or reliability. A far score can still be wrong on an unfamiliar scanner or population."],
        ["MC-dropout variation", "Shows how much repeated dropout predictions vary.", "It is exploratory score variation, not calibrated diagnostic uncertainty."],
        ["MRI slice viewer", "Lets the reviewer inspect original slices by plane.", "This is for research review and does not replace a radiology viewer."],
        ["Model attention", "Shows original slice beside Grad-CAM for one target, one plane and one sampled slice.", "It deliberately avoids averaging across planes or manually intensifying colour because those could imply false localisation."],
        ["Performance graphs", "Displays recorded ROC, precision-recall and confusion-matrix evidence.", "These are development results, not clinical acceptance evidence."],
        ["Export and clear", "Exports a result summary and clears the browser-session case.", "Export aids research traceability; clear case prevents an earlier case from remaining on screen."],
    ], [1.5, 2.4, 3.2])

    add_heading(doc, "6. How to explain the measured results", 1)
    doc.add_paragraph("Use these exact numbers only with their dataset and limitation. They are from the selected calibrated checkpoint on the 120-study MRNet official-validation partition, which had already been used during development.")
    add_table(doc, ["Finding", "AUROC", "F1", "Sensitivity", "Specificity", "Safe conclusion"], [
        ["General abnormality", "0.901", "0.927", "0.937", "0.680", "The strongest result; supports the primary research task on this MRNet cohort."],
        ["ACL tear", "0.825", "0.746", "0.815", "0.697", "Promising internal evidence, but insufficient for clinical use."],
        ["Meniscal tear", "0.754", "0.631", "0.673", "0.647", "The weakest target; present it as an additional exploratory output."],
        ["Macro average", "0.827", "0.768", "0.808", "0.675", "A summary across three different labels; not a universal patient-level accuracy."],
    ], [1.35, .95, .65, 1.05, 1.05, 2.45])
    add_bullets(doc, [
        "AUROC: how well the model ranks positive above negative cases across possible thresholds. It does not choose a clinical action threshold.",
        "Sensitivity: proportion of labelled positive studies flagged by the model. Missed positives are false negatives.",
        "Specificity: proportion of labelled negative studies not flagged. Incorrect flags are false positives.",
        "Precision: among flagged studies, the fraction with the reference positive label in that cohort.",
        "F1-score: balance of precision and sensitivity at one selected research threshold.",
        "Accuracy: proportion correct at that threshold. It can be misleading when the class balance changes.",
    ])
    doc.add_paragraph("External evidence was weaker: KneeMRI ACL AUROC was 0.603 across 909 exam groups; the limited fastMRI meniscus reference evaluation was 0.656 across 199 files. These show that generalisation is not established.")

    add_heading(doc, "7. Grad-CAM: answer this carefully", 1)
    add_table(doc, ["Question", "Accurate answer"], [
        ["What does the heatmap show?", "Warm pixels show areas that positively contributed to the selected model score for one target, plane and sampled slice."],
        ["Why is it coarse?", "EfficientNet produces a low-resolution final feature map, which is enlarged to the 224 × 224 display. The model also has study-level labels, not lesion locations."],
        ["Does it show the tear?", "No. It does not confirm a tear, identify tissue, measure lesion size, define a boundary or rule out pathology elsewhere."],
        ["Why is it sometimes central or broad?", "The model can use general knee context because it was trained only to classify the whole study. Without boxes or masks, there is no training signal forcing attention onto the exact lesion."],
        ["What would validate localisation?", "A separate dataset with radiologist-verified lesion boxes or segmentation masks, followed by measures such as overlap or pointing accuracy."],
    ], [2.1, 4.9])

    add_heading(doc, "8. Likely defence questions and safe answers", 1)
    questions = [
        ("Why did you choose MRI?", "MRI is the modality represented in the training data and is suitable for soft-tissue knee structures. The project does not interpret X-rays or photographs."),
        ("Why study-level classification instead of one image?", "A knee MRI examination is a series of slices. The model samples and aggregates slices so the output represents the study rather than one arbitrary image."),
        ("Why EfficientNet-B0?", "It gave the stronger recorded internal-selection macro AUROC when compared with ResNet-18, while remaining computationally manageable for a fourth-year prototype."),
        ("Did you train five models?", "A separate five-family benchmark was completed, but its best new EfficientNet-B0 candidate did not beat the active calibrated baseline. I did not promote it or claim an ensemble."),
        ("Why is the encoder pretrained on ImageNet?", "Transfer learning starts with generic visual features, then the classification head is fitted using knee MRI labels. It improves practicality with limited public MRI data."),
        ("How do you prevent data leakage?", "Fitting, tuning, calibration and validation lists are separated at study level, and I perform exact-series checks. Patient linkage was unavailable, so patient-level independence is a stated limitation."),
        ("What does a 90.1% AUROC mean?", "It means the model ranked positive general-abnormality studies above negative studies well on that MRNet cohort. It does not mean 90.1% diagnostic accuracy or clinical approval."),
        ("Why is specificity lower than sensitivity for general abnormality?", "The selected threshold prioritised catching labelled abnormal studies, which increased false-positive flags. This trade-off is visible in the confusion matrix and is not a clinical operating rule."),
        ("Can the system diagnose an ACL tear?", "No. It produces a research score for ACL tear. A radiologist or qualified clinician remains responsible for interpretation."),
        ("Why are external results weaker?", "The external data differ in acquisition, labels and available planes. This domain shift is evidence that MRNet-only training does not prove generalisation."),
        ("Is Grad-CAM a detection method?", "No. It explains a classifier score. Detection needs labelled bounding boxes; segmentation needs labelled masks. Neither is implemented or claimed."),
        ("Why reject JPEG or PNG images?", "A single screenshot loses the study structure and differs from the MRI volumes used in training. Accepting it would create unsupported predictions."),
        ("What does the dashboard model-selection panel do?", "It checks the registry for compatible prevalidated routes and shows their evidence. It assigns DenseNet-121 to general abnormality, ResNet-18 to ACL and Swin-T to meniscus before scoring, without using patient scores to choose a model."),
        ("What are the project limitations?", "MRNet-only supervised training, no patient linkage, one seed, reused development evaluation, limited external references, no clinical reader study, no lesion labels and no clinical validation."),
        ("What would you do next?", "Lock the model and thresholds, obtain an independent patient-level multi-site dataset with qualified reference labels, evaluate calibration and subgroups, then conduct reader and workflow studies."),
    ]
    for question, answer in questions:
        add_qa(doc, question, answer)

    add_heading(doc, "9. Two-minute live demonstration script", 1)
    add_bullets(doc, [
        "Start: “This is a research interface for a complete knee MRI study, not a diagnosis screen.”",
        "Show the supported input list and say why screenshots are rejected.",
        "Show study intake: explain file checks, plane metadata and manual confirmation.",
        "Click Analyse study: explain preprocessing, sampled slices, EfficientNet-B0 and study-level aggregation.",
        "Point to each score and threshold: “Flagged for research review is not a confirmed finding.”",
        "Open the slice viewer: “This is the original sampled MRI slice.”",
        "Open model attention: “This explains score contribution on one plane and slice. It is not a tear outline.”",
        "Open performance figures: state the MRNet values and immediately state the external/generalisation limitation.",
        "Show export and clear-case controls: explain traceability and preventing stale results from a prior case.",
    ])

    add_heading(doc, "10. Final speaking rules", 1)
    add_bullets(doc, [
        "Say “research prototype”, “study-level score”, “reference label”, “flagged for research review” and “model attention”.",
        "Do not say “diagnoses”, “detects tears”, “shows the lesion”, “clinically accurate”, “high confidence” or “ready for hospitals”.",
        "If challenged, state the limitation first, then describe the evidence you do have. Honest boundaries make the project stronger.",
        "Only say that you implemented, trained, tested or wrote an item if you can open it, explain it and reproduce the relevant step during the defence.",
    ])
    doc.add_paragraph("Evidence locations: `runs/mri_finetune_02/validation_summary.json`, `model_registry/active_models.json`, `results/`, `src/inference/predictor.py`, `src/evaluation/gradcam.py`, `app.py`, and `README.md`.").italic = True

    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
