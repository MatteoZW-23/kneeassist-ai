import hashlib
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from src.utils import ROOT,config,resolve
from src.data.preprocessing import load_uploads,InputError
from src.routing.smart_input_router import inspect_uploads
from src.selection.model_selector import select_models
from src.selection.evidence import candidate_evidence, display_name, formal_comparison_evidence, objective_traceability, ranking_for_target, split_summary
from src.selection.ensemble import ensemble_plans
from src.localization.registry import localization_status
from src.inference.predictor import text_summary
from src.inference.routed_predictor import RoutedPredictor
from src.inference.dynamic_predictor import DynamicSelectionPredictor, DynamicSelectionUnavailable, dynamic_selection_status
from src.evaluation.gradcam import is_valid_attention_payload

st.set_page_config(page_title='KneeAssist XAI · MRI review',page_icon='🩻',layout='wide')
cfg=config()
if not cfg.get('lifecycle',{}).get('inference_enabled',True):
    st.title('KneeAssist XAI')
    st.info('MRI analysis is disabled in the current deployment configuration.')
    st.caption('Enable inference in config.yaml and relaunch the application.')
    st.stop()
st.markdown('''<style>
.stApp {background:#f5f8fc;}
[data-testid="stSidebar"] {background:#102b3f;}
[data-testid="stSidebar"] * {color:#f1f7fa;}
[data-testid="stSidebar"] button p {color:#12344b;}
h1,h2,h3 {color:#12344b;}
[data-testid="stMetric"] {background:white;border:1px solid #dce6ee;border-radius:12px;padding:18px;}
.intro {background:linear-gradient(110deg,#11394c,#17697a);padding:24px 28px;border-radius:14px;color:white;margin-bottom:18px;}
.intro h1 {color:white;margin:0;font-size:2.1rem;}.intro p {color:#d8eef2;margin:6px 0 0;}
.tag {font-size:.8rem;letter-spacing:.12em;text-transform:uppercase;color:#aee3df;}
</style>''',unsafe_allow_html=True)

def routing_stamp():
    """Invalidate the cached predictor when its registry, checkpoints or code changes."""
    registry_path = ROOT / 'model_registry' / 'active_models.json'
    if not registry_path.exists():
        return ('registry-unavailable',)
    registry = json.loads(registry_path.read_text(encoding='utf-8'))
    entries = []
    for model in registry.get('models', []):
        checkpoint = ROOT / model['checkpoint']
        entries.append((model['id'], str(checkpoint), checkpoint.stat().st_mtime_ns, checkpoint.stat().st_size)
                       if checkpoint.exists() else (model['id'], str(checkpoint), 'missing'))
    dynamic_path = ROOT / cfg.get('dynamic_selection', {}).get('artifact', '')
    implementation_files = (
        ROOT / 'src' / 'inference' / 'predictor.py',
        ROOT / 'src' / 'inference' / 'routed_predictor.py',
        ROOT / 'src' / 'inference' / 'dynamic_predictor.py',
        ROOT / 'src' / 'dynamic_selection' / 'selector.py',
    )
    implementation_stamp = tuple(
        (str(path), path.stat().st_mtime_ns if path.exists() else 'missing')
        for path in implementation_files
    )
    dynamic_stamp = (str(dynamic_path), dynamic_path.stat().st_mtime_ns, dynamic_path.stat().st_size) if dynamic_path.is_file() else (str(dynamic_path), 'missing')
    dynamic_status=dynamic_selection_status(cfg)
    dynamic_checkpoints=tuple(dynamic_status.get('checkpoint_stamps',[]))
    return (registry_path.stat().st_mtime_ns, tuple(entries), implementation_stamp, dynamic_stamp,
            bool(cfg.get('dynamic_selection', {}).get('enabled', False)),dynamic_checkpoints,dynamic_status.get('reason'))

@st.cache_resource
def cached_predictor(registry_version_stamp):
    status = dynamic_selection_status(cfg)
    if status['available']:
        return DynamicSelectionPredictor(cfg=cfg)
    if cfg.get('dynamic_selection',{}).get('enabled',False):
        raise DynamicSelectionUnavailable('Dynamic inference is unavailable: '+status['reason'])
    return RoutedPredictor(cfg=cfg)

def get_predictor():
    return cached_predictor(routing_stamp())


def display_architecture(architecture):
    """Use familiar model names wherever a registered architecture is shown."""
    names = {
        'densenet121': 'DenseNet-121',
        'resnet18': 'ResNet-18',
        'resnet50': 'ResNet-50',
        'efficientnet_b0': 'EfficientNet-B0',
        'swin_t': 'Swin Transformer (Swin-T)',
    }
    return names.get(str(architecture).lower(), str(architecture))


def _selection_table(target):
    """Format saved candidate evidence for a transparent, auditable table."""
    rows = []
    for rank, item in enumerate(ranking_for_target(target), start=1):
        metrics = item['metrics']
        rows.append({
            'Rank': rank,
            'Model': item['model'],
            'Selection score': round(item['selection_score'], 3),
            'AUROC': round(metrics['auroc'], 3),
            'PR-AUC (AP)': round(metrics['average_precision'], 3),
            'Sensitivity': round(metrics['sensitivity'], 3),
            'Specificity': round(metrics['specificity'], 3),
            'Precision': round(metrics['precision'], 3),
            'F1': round(metrics['f1'], 3),
            'Accuracy': round(metrics['accuracy'], 3),
            'Brier score': round(item['brier'], 3) if item['brier'] is not None else 'Not retained',
            'Inference time': (f"{item['inference_time']:.3f} s" if item['inference_time'] is not None else 'Not retained'),
            'Archived fixed-route winner': 'Yes' if rank == 1 else 'No',
            'Checkpoint status': item['status'],
        })
    return pd.DataFrame(rows)


def _render_saved_candidate_artifacts(target, rankings):
    """Show retained, real benchmark figures without recomputing model results."""
    labels = {item['model']: item['architecture'] for item in rankings}
    selected_name = st.selectbox(
        f'Show retained figures for {target}',
        list(labels),
        key=f'candidate_artifact_{target}',
    )
    architecture = labels[selected_name]
    folder = ROOT / 'runs' / 'model_family_v2' / 'results' / architecture
    figures = [
        ('ROC curves', folder / 'roc_curves.png'),
        ('Precision-recall curves', folder / 'precision_recall_curves.png'),
        ('Confusion matrices', folder / 'confusion_matrices.png'),
    ]
    available = [(title, path) for title, path in figures if path.exists()]
    if available:
        cols = st.columns(len(available))
        for column, (title, path) in zip(cols, available):
            with column:
                st.caption(title)
                st.image(str(path), width='stretch')
    else:
        st.info('No retained figure is available for this candidate. The metric table remains the source of recorded evidence.')


def _dynamic_experiment_evidence():
    """Load a completed dynamic-selection evaluation without manufacturing values."""
    path = ROOT / 'runs' / 'dynamic_selection_v1' / 'final_evaluation.json'
    if not path.is_file():
        return None, 'The five-model dynamic-selection experiment has not completed yet.'
    try:
        evidence = json.loads(path.read_text(encoding='utf-8'))
        required = {'base_models', 'dynamic_selection', 'selection_frequency', 'activation_status'}
        missing = required - set(evidence)
        if missing:
            return None, 'The dynamic-selection evaluation is incomplete: missing ' + ', '.join(sorted(missing)) + '.'
        return evidence, None
    except (OSError, json.JSONDecodeError) as error:
        return None, f'The dynamic-selection evaluation could not be read: {error}'


def _dynamic_metric_table(evidence, target):
    """Show only measured final-partition metrics for DMS and its base models."""
    frequency = evidence['selection_frequency']
    rows = []
    for model_id, report in evidence['base_models'].items():
        metrics = report.get(target, {})
        architecture = model_id.replace('dms_', '')
        rows.append({
            'Approach': display_architecture(architecture),
            'AUROC': metrics.get('auroc'),
            'PR-AUC (AP)': metrics.get('average_precision'),
            'Sensitivity': metrics.get('sensitivity'),
            'Specificity': metrics.get('specificity'),
            'Precision': metrics.get('precision'),
            'F1': metrics.get('f1'),
            'Accuracy': metrics.get('accuracy'),
            'Brier score': metrics.get('brier'),
            'ECE (10 bins)': metrics.get('ece_10_equal_width_bins'),
            'Log loss': metrics.get('log_loss'),
            'Final-partition selections': frequency.get(model_id, 0),
            'Role': 'Base classifier',
        })
    dms_metrics = evidence['dynamic_selection'].get(target, {})
    rows.append({
        'Approach': 'Dynamic selector (case-specific)',
        'AUROC': dms_metrics.get('auroc'),
        'PR-AUC (AP)': dms_metrics.get('average_precision'),
        'Sensitivity': dms_metrics.get('sensitivity'),
        'Specificity': dms_metrics.get('specificity'),
        'Precision': dms_metrics.get('precision'),
        'F1': dms_metrics.get('f1'),
        'Accuracy': dms_metrics.get('accuracy'),
        'Brier score': dms_metrics.get('brier'),
        'ECE (10 bins)': dms_metrics.get('ece_10_equal_width_bins'),
        'Log loss': dms_metrics.get('log_loss'),
        'Final-partition selections': 'Study-specific',
        'Role': 'Proposed system',
    })
    return pd.DataFrame(rows)


def render_dynamic_selection_evidence():
    """Render the independently trained five-model case-specific selector evidence."""
    evidence, reason = _dynamic_experiment_evidence()
    st.subheader('Dynamic model-selection experiment')
    if evidence is None:
        st.info(reason)
        return
    status = dynamic_selection_status(cfg)
    st.caption(
        'For each study, the five base models run first. A Gradient Boosting competence estimator then chooses the architecture '
        'with the lowest estimated mean Brier loss from current study features and five-model outputs. It cannot access an uploaded study label.'
    )
    if status['available']:
        st.success('The verified selector is active for new MRI analyses.')
    else:
        st.warning('The experiment is recorded, but the selector is not active for new MRI analyses: ' + status['reason'])
    st.caption('Evaluation partition: ' + str(evidence.get('partition', 'not recorded')) + '. ' + str(evidence.get('limitation', '')))
    st.caption(
        f"Selector varied across {sum(1 for count in evidence['selection_frequency'].values() if count)} of 5 candidates on the final partition. "
        'Selection frequency describes recorded decisions; it is not a global ranking.'
    )
    latency = evidence.get('latency')
    if latency:
        seconds = latency.get('seconds', {})
        st.caption(
            f"Measured local end-to-end dynamic inference: mean {seconds.get('mean', float('nan')):.2f} s "
            f"over {latency.get('repeat_count', '—')} repeat(s), after warm-up. " + str(latency.get('limitation', ''))
        )
    st.dataframe(
        pd.DataFrame([
            {'Model': display_architecture(model_id.replace('dms_', '')), 'Final-partition selections': count}
            for model_id, count in evidence['selection_frequency'].items()
        ]),
        hide_index=True,
        width='stretch',
    )
    macro_table = _dynamic_metric_table(evidence, 'macro')
    st.subheader('Macro comparison and calibration')
    st.dataframe(
        macro_table[['Approach', 'AUROC', 'PR-AUC (AP)', 'F1', 'Brier score', 'ECE (10 bins)', 'Log loss', 'Role']].round(3),
        hide_index=True,
        width='stretch',
    )
    st.caption(
        'Lower Brier score, ECE and log loss indicate better probability quality. The displayed ECE uses ten equal-width bins; '
        'it is descriptive for this recorded partition, not clinical calibration evidence.'
    )
    figures_dir = ROOT / 'runs' / 'dynamic_selection_v1' / 'figures'
    figure_columns = st.columns(2)
    for column, filename, caption in (
        (figure_columns[0], 'model_comparison.png', 'Macro metric comparison'),
        (figure_columns[1], 'calibration_curves.png', 'Calibration curves on the recorded final partition'),
    ):
        figure = figures_dir / filename
        if figure.is_file():
            with column:
                st.image(str(figure), caption=caption, width='stretch')
    st.caption(str(evidence.get('comparison_conclusion', '')))
    tabs = st.tabs([target['display'] for target in cfg['targets']])
    for tab, target in zip(tabs, cfg['targets']):
        with tab:
            table = _dynamic_metric_table(evidence, target['key'])
            st.dataframe(table.round(3), hide_index=True, width='stretch')
            chart = table.set_index('Approach')[['AUROC', 'PR-AUC (AP)', 'Sensitivity', 'Specificity', 'F1']]
            st.bar_chart(chart, width='stretch')
            st.caption(
                'All six rows use the same recorded final studies. The dynamic row changes architecture per study; it is not a fixed model. '
                'Final-partition labels were excluded from selector training and are used here only for evaluation.'
            )


def render_recorded_model_evaluation():
    """Read-only evidence page for the formal study and supplementary portfolio."""
    evidence = candidate_evidence()
    weights = evidence['weights']
    st.caption('Recorded baseline experiments and static-routing evidence. This section does not describe the active dynamic selector.')
    st.warning('All values on this page are MRNet development evidence. The 120-study official-validation cohort had previous development exposure and is not an independent clinical test.')

    formal = formal_comparison_evidence()
    st.subheader('Recorded ResNet-18 and EfficientNet-B0 comparison')
    if formal.get('available'):
        st.success(f"Selected model: {formal['winner']} — highest saved internal-validation macro AUROC in the ResNet-18 versus EfficientNet-B0 study.")
        st.dataframe(pd.DataFrame(formal['rows']).round(3), hide_index=True, width='stretch')
        counts = formal.get('counts', {})
        st.caption(
            f"Fixed study roles: {counts.get('fit', '—')} fitting, {counts.get('tune', '—')} tuning, "
            f"{counts.get('calibration', '—')} calibration and {counts.get('official_valid', '—')} post-selection development evaluation studies. "
            "The result is descriptive because one repeat was retained."
        )
        st.caption(formal['limitation'])
    else:
        st.info(formal.get('reason', 'Formal comparison evidence is unavailable.'))

    st.divider()
    st.subheader('Archived fixed-routing benchmark — not used for uploaded studies')
    st.caption('This retained benchmark documents an earlier fixed target-routing comparison. It is not used while the active dynamic selector is enabled for uploaded studies.')

    st.subheader('Archived benchmark design')
    design_left, design_right = st.columns([1, 1.25])
    with design_left:
        st.dataframe(pd.DataFrame(split_summary()), hide_index=True, width='stretch')
    with design_right:
        st.markdown('**Candidate architectures**')
        st.write('ResNet-18, ResNet-50, DenseNet-121, EfficientNet-B0 and Swin Transformer (Swin-T).')
        st.markdown('**Controlled conditions**')
        st.caption('The five candidates use the same saved MRI study IDs, target labels, 224 × 224 preprocessing, 12 sampled slices per plane, study-level mean/max aggregation and locked 226-study tuning partition.')
        st.markdown('**Selection score**')
        formula = ' + '.join(f"{float(weight):.0%} × {name.replace('_', ' ').upper()}" for name, weight in weights.items())
        st.code('Selection score = ' + formula, language=None)
        st.caption('Calibration is reported with Brier score and log loss. It is not included in the score because ECE was not retained; no value is invented.')

    st.subheader('Archived fixed-route ranking from saved validation evidence')
    tabs = st.tabs([target['display'] for target in cfg['targets']])
    for tab, target in zip(tabs, cfg['targets']):
        with tab:
            ranking = ranking_for_target(target['key'])
            table = _selection_table(target['key'])
            if not ranking:
                st.error('No saved candidate evidence was found for this target.')
                continue
            winner = ranking[0]
            st.info(f"Archived fixed-route winner: {winner['model']} (selection score {winner['selection_score']:.3f}). This saved comparison is shown for research history only; it is not the active upload-time dynamic selection decision.")
            st.dataframe(table, hide_index=True, width='stretch')
            chart = table.set_index('Model')[['AUROC', 'PR-AUC (AP)', 'Sensitivity', 'Specificity', 'F1', 'Selection score']]
            st.bar_chart(chart, width='stretch')
            _render_saved_candidate_artifacts(target['key'], ranking)
            st.caption('This is retained benchmark evidence. The live dynamic workflow instead executes all five candidates and uses the saved expected-loss selector for the uploaded study.')

    st.subheader('Archived fixed routes — dynamic selection fallback only')
    route_rows = []
    for target in cfg['targets']:
        ranked = ranking_for_target(target['key'])
        winner = ranked[0] if ranked else None
        if winner:
            route_rows.append({
                'Finding': target['display'],
                'Selected model': winner['model'],
                'Selection score': round(winner['selection_score'], 3),
                'Selection basis': 'Highest predefined validation score among compatible benchmark candidates',
                'Fallback availability': 'Available if dynamic selection is disabled' if winner['locally_available'] else 'Unavailable',
            })
    st.dataframe(pd.DataFrame(route_rows), hide_index=True, width='stretch')
    st.caption('These routes are used only if dynamic selection is deliberately disabled. Under the normal active configuration, the system runs all five candidates and selects one architecture through the saved learned selector for each complete study.')

    st.subheader('Research traceability')
    st.dataframe(pd.DataFrame(objective_traceability()), hide_index=True, width='stretch')
    st.info('Detection boxes and segmentation masks are not enabled because no trained and validated lesion-location or segmentation model is registered. Grad-CAM is available as coarse classifier attention only.')


def render_model_evaluation_page():
    st.markdown('<div class="intro"><div class="tag">Measured model evidence</div><h1>Model evaluation and selection</h1><p>Compare five trained classifiers with the learned study-specific selection system.</p></div>',unsafe_allow_html=True)
    render_dynamic_selection_evidence()


def render_research_workflow_page():
    st.markdown('<div class="intro"><div class="tag">Reproducible research workflow</div><h1>Research Workflow</h1><p>This page summarises the offline model-development workflow and the local MRI-analysis process implemented in this application.</p></div>', unsafe_allow_html=True)
    st.subheader('Workflow stages')
    if cfg.get('dynamic_selection',{}).get('enabled',False):
        steps = [
            ('1','Review the fixed MRNet roles','The experiment uses 704 fitting, 100 tuning, 100 calibration, 226 selector-training and 120 final-evaluation studies. Study and exact-series overlaps are checked; patient linkage is unavailable.'),
            ('2','Train five base classifiers','ResNet-18, ResNet-50, DenseNet-121, EfficientNet-B0 and Swin-T use the same study IDs, preprocessing and three target labels. This run trains classifier heads with frozen ImageNet encoders.'),
            ('3','Calibrate outputs','Each base model is calibrated only on the reserved calibration partition.'),
            ('4','Train the competence estimators','Five Gradient Boosting competence estimators learn from model-output and study features on the separate 226-study selector partition. Ground truth supplies mean Brier-loss training targets only during offline development.'),
            ('5','Evaluate the complete system','All five bases and dynamic selection are scored on the same 120 studies excluded from DMS fitting. Prior project exposure prevents an independent-test claim.'),
            ('6','Upload an MRI study','Binary content, dimensions, finite intensities and available metadata are checked. Processed arrays with unknown planes require manual confirmation.'),
            ('7','Run all five models','Every valid analysis executes every candidate on the same original study, with the same normalisation, resizing and slice sampling.'),
            ('8','Select for this study','Current probabilities, entropy, disagreement and acquisition features enter the learned selector. The architecture with the lowest estimated mean Brier loss provides all three final outputs.'),
            ('9','Review the decision','The application shows five-model probabilities, estimated losses, selected architecture, thresholds and timing. A new upload invalidates the previous result.'),
            ('10','View attention and export','Grad-CAM uses the selected model. Its coarse classifier-attention overlay and the reproducible analysis summary can be exported. Detection and segmentation remain unavailable.'),
        ]
    else:
        steps = [
        ('1', 'Review the fixed MRNet split', 'The Fixed experimental design table records fitting, tuning validation, calibration and official-validation roles.'),
        ('2', 'Review the primary comparison', 'The Model evidence page lists the saved ResNet-18 and EfficientNet-B0 comparison used in the formal study.'),
        ('3', 'Review supplementary routing evidence', 'The same page lists the separate target-specific portfolio, candidate ranking and saved artefacts.'),
        ('4', 'Review the selection policy', 'The predefined weighted score uses tuning validation; case probabilities never choose the model.'),
        ('5', 'Review the registered routes', 'DenseNet-121 is registered for general abnormality, ResNet-18 for ACL tear and Swin-T for meniscal tear.'),
        ('6', 'Use Clinical MRI review', 'Upload one supported MRI study or use an included research example.'),
        ('7', 'Review file intake', 'The application reports file type, dimensions and plane confirmation. Names such as 0000.npy are not trusted as plane labels.'),
        ('8', 'Review registry retrieval', 'Model selection and study compatibility lists the selected route and saved validation rationale for every target.'),
        ('9', 'Review study-level scores', 'The application reports three research probabilities, thresholds and models used; these are not diagnoses.'),
        ('10', 'Review attention and limits', 'Grad-CAM shows coarse positive classifier attention for a selected target and slice, not a lesion boundary, detection box or segmentation mask.'),
        ]
    for number, title, detail in steps:
        st.markdown(f'**Stage {number}: {title}.** {detail}')
    st.subheader('Read-only evidence check')
    st.code('.\\Show Research Evidence.cmd', language='powershell')
    st.caption('The command opens the saved formal comparison and supplementary routing evidence without training, changing models or scoring a new study.')
    st.subheader('Research limitations')
    st.warning('The system is a research decision-support prototype trained on MRNet labels. It has no independent clinical validation, patient-level linkage verification, radiologist reader study, validated detector, segmentation model or treatment recommendation function.')

if 'upload_generation' not in st.session_state:st.session_state.upload_generation=0

def clear_case():
    generation=st.session_state.upload_generation+1
    st.session_state.clear()
    st.session_state.upload_generation=generation

with st.sidebar:
    st.markdown('## KneeAssist XAI')
    st.caption('MRI RESEARCH WORKSPACE · V1')
    st.divider()
    workspace = st.radio(
        'Workspace',
        ['Clinical MRI review', 'Model evaluation and selection', 'Research workflow'],
        key='workspace_navigation',
    )
    st.divider()
    st.markdown('**Study-level review**\n\n'+'\n\n'.join(t['display'] for t in cfg['targets']))
    st.divider()
    st.caption('Runs locally. Uploaded images and case references remain in this browser session and are not written to application logs.')
    st.button('Clear current case',width='stretch',on_click=clear_case)
    st.caption('Research and clinical decision support. Not an autonomous diagnostic system.')

if workspace == 'Model evaluation and selection':
    render_model_evaluation_page()
    st.stop()
if workspace == 'Research workflow':
    render_research_workflow_page()
    st.stop()

st.markdown('<div class="intro"><div class="tag">Knee MRI · explainable research review</div><h1>KneeAssist XAI</h1><p>Review study-level model scores, model selection and attention maps. Clinical judgment remains essential.</p></div>',unsafe_allow_html=True)
st.warning('Research decision-support prototype. Clinical review is required. Model attention is not a confirmed lesion or lesion location.')
with st.expander('Model evidence and supported inputs'):
    from src.inference.readiness import deployment_evidence
    evidence=deployment_evidence(cfg)
    dynamic_status=dynamic_selection_status(cfg)
    dynamic_evidence,_=_dynamic_experiment_evidence()
    if cfg.get('dynamic_selection',{}).get('enabled',False):
        st.caption('Active analysis: all-five-model inference followed by learned case-specific competence selection.' if dynamic_status['available'] else 'Dynamic inference unavailable: '+dynamic_status['reason'])
        if dynamic_status.get('available'):
            st.caption('Dynamic selection accepts complete axial, coronal and sagittal studies only. It does not select a model for a missing-plane study because that use was not evaluated.')
            for warning in dynamic_status.get('provenance_warnings',[]):
                st.warning(warning)
        evidence={'metrics':dynamic_evidence['dynamic_selection'] if dynamic_evidence else None}
    else:
        st.caption('Active registered research models: '+evidence['checkpoint'])
    st.markdown('**Use one knee MRI study at a time.** All uploaded sequences must belong to the same case. File checks cannot confirm anatomy, patient identity or acquisition suitability.')
    st.caption('MRI NumPy stacks, NIfTI volumes and DICOM studies are inspected before analysis. Unlabelled processed arrays require manual plane confirmation. JPEG/PNG are not supported. No finding rules out an injury.')
    if evidence['metrics']:
        st.caption('Previously evaluated MRNet validation cohort. These measurements are not clinical acceptance criteria.')
        st.dataframe(pd.DataFrame([{'Finding':t['display'],'AUROC':round(evidence['metrics'][t['key']]['auroc'],3),
                       'Sensitivity':round(evidence['metrics'][t['key']]['sensitivity'],3),
                       'Specificity':round(evidence['metrics'][t['key']]['specificity'],3)} for t in cfg['targets']]),hide_index=True)
    st.caption('Training and reported evaluation use MRNet development data. Patient independence and clinical confidence are not established.')
left,right=st.columns([1.05,1.8],gap='large')
with left:
    st.subheader('1 · Add a study')
    case=st.text_input('Case reference',placeholder='e.g. RESEARCH-001',max_chars=120,key=f'case_reference_{st.session_state.upload_generation}')
    uploads=st.file_uploader('MRI study files',type=['npy','npz','zip','dcm','nii','gz'],accept_multiple_files=True,
        help='Use .npy, .npz, .nii/.nii.gz, .dcm, or ZIP study files. Binary content and MRI structure are inspected before prediction.',
        key=f'uploads_{st.session_state.upload_generation}')
    st.caption('Supported: MRI NumPy stacks/NPZ archives, NIfTI volumes, DICOM slices and study ZIPs. Dynamic selection requires axial, coronal and sagittal sequences from the same study. File checks cannot confirm knee anatomy or clinical suitability. JPEG/PNG are rejected.')
    st.caption('Included research examples contain three-plane MRNet validation studies: axial, coronal and sagittal.')
    if st.button('Load research example',width='stretch'):
        demo=ROOT/'sample_cases/mrnet_1130.zip'
        if uploads:
            st.info('Clear the current case before loading the packaged example.')
        elif demo.exists():
            st.session_state.demo=('mrnet_1130.zip',demo.read_bytes());st.session_state.pop('result',None)
        else:st.error('The packaged example is unavailable. Upload a study instead.')
    if st.button('Load abnormal research example',width='stretch'):
        demo=ROOT/'sample_cases/mrnet_1229_abnormal_research_example.zip'
        if uploads:
            st.info('Clear the current case before loading the packaged example.')
        elif demo.exists():
            st.session_state.demo=('mrnet_1229_abnormal_research_example.zip',demo.read_bytes());st.session_state.pop('result',None)
        else:st.error('The packaged example is unavailable. Upload a study instead.')
    if uploads:st.session_state.pop('demo',None)
    elif 'demo' in st.session_state:
        demo_name=st.session_state.demo[0]
        demo_description='normal-reference' if demo_name=='mrnet_1130.zip' else 'abnormal-reference'
        st.info(f'Research example loaded: MRNet validation study ({demo_description}); three MRI planes included.')
    files=[(u.name,u.getvalue()) for u in uploads] if uploads else ([st.session_state.demo] if 'demo' in st.session_state else [])
    intake=None;plane_overrides={}
    if files:
        try:
            intake=inspect_uploads(files,cfg['preprocessing'])
            with st.expander('Study intake and plane confirmation',expanded=True):
                intake_rows=[]
                for item in intake['files']:
                    detected=item['detected_plane'].title() if item['detected_plane'] else 'Unknown'
                    confidence='Not available' if item['plane_confidence'] is None else f"{item['plane_confidence']:.0%}"
                    shape=' × '.join(map(str,item['shape'])) if item['shape'] else 'DICOM metadata only'
                    intake_rows.append({'File':item['file'],'Type':item['format'],'Shape':shape,'Detected plane':detected,'Plane confidence':confidence,'Status':item['status']})
                st.dataframe(pd.DataFrame(intake_rows),hide_index=True,width='stretch')
                for confirmation_index,item in enumerate(intake['requires_confirmation']):
                    # The uploader can contain files with the same display name.  Include
                    # their position so Streamlit never reuses a widget key; the intake
                    # identifier remains the key passed to the file loader.
                    widget_key='plane_'+hashlib.sha256(item['id'].encode()).hexdigest()[:12]+f'_{confirmation_index}'
                    choice=st.selectbox(f"Confirm plane for {item['file']}",['Choose plane','axial','coronal','sagittal'],key=widget_key)
                    if choice!='Choose plane':plane_overrides[item['id']]=choice
                if intake['requires_confirmation']:
                    st.caption('Manual confirmation is required because this input lacks reliable anatomical orientation metadata. A filename alone is not treated as proof of plane.')
        except InputError as e:
            st.error(f'Input inspection failed: {e}')
    model_stamp=routing_stamp()
    current_signature=hashlib.sha256(b''.join(n.encode()+b for n,b in files)+case.encode()+json.dumps(plane_overrides,sort_keys=True).encode()+repr(model_stamp).encode()).hexdigest()
    # Never show a previous case's predictions against changed inputs.
    if st.session_state.get('case_signature')!=current_signature:
        for key in ['result','volumes','attention_cache']:st.session_state.pop(key,None)
        for key in list(st.session_state):
            if key.startswith('cam_slice_'):st.session_state.pop(key,None)
    if st.button('Analyse study',type='primary',width='stretch'):
        try:
            if sum(len(b) for _,b in files)>cfg['preprocessing']['max_upload_mb']*1024**2:
                raise InputError('The combined study files exceed the 100 MB upload limit.')
            if not files:
                raise InputError('Upload at least one MRI sequence before analysing the case.')
            if intake is None:
                raise InputError('The uploaded files could not be inspected. Correct the input error and try again.')
            unconfirmed=[item['file'] for item in intake['requires_confirmation'] if item['id'] not in plane_overrides]
            if unconfirmed:
                raise InputError('Confirm the MRI plane for: '+', '.join(unconfirmed)+'.')
            progress=st.progress(0,text='Checking MRI files…')
            volumes=load_uploads(files,cfg['preprocessing'],plane_overrides)
            dynamic_status=dynamic_selection_status(cfg)
            if cfg.get('dynamic_selection',{}).get('enabled',False) and not dynamic_status['available']:
                raise InputError('Dynamic inference is unavailable: '+dynamic_status['reason'])
            ensemble_plan=ensemble_plans(list(volumes),[target['key'] for target in cfg['targets']])
            localizers=localization_status()
            predictor=get_predictor()
            if dynamic_status['available']:
                progress.progress(25,text='Preparing all five trained candidates…')
                progress.progress(40,text='Running five model inferences and case-specific selection…')
                with st.spinner('Analysing this MRI study with all five candidate models…'):
                    result=predictor.predict(volumes, case or 'Not supplied')
                dynamic=result['dynamic_selection']
                selection={
                    'mode':'dynamic_case_specific_competence_estimation',
                    'available_planes':sorted(volumes),
                    'models_evaluated':dynamic['models_evaluated'],
                    'selected_model_id':dynamic['selected_model_id'],
                    'selected_architecture':dynamic['selected_architecture'],
                    'expected_losses':dynamic['expected_losses'],
                    'base_probabilities':dynamic['base_probabilities'],
                    'total_seconds':dynamic['total_seconds'],
                    'selection_rule':dynamic['selection_rule'],
                    'principal_factors':dynamic.get('principal_factors',[]),
                    'factor_interpretation':dynamic.get('factor_interpretation'),
                    'selector_estimator':dynamic.get('selector_estimator'),
                    'selector_provenance_warnings':dynamic.get('selector_provenance_warnings',[]),
                    'interpretation':dynamic['interpretation'],
                }
            else:
                progress.progress(25,text='Comparing compatible validated model candidates…')
                selection=select_models(list(volumes),[target['key'] for target in cfg['targets']])
                unavailable=[entry['target'] for entry in selection['selections'] if entry['status']!='Selected']
                if unavailable:
                    raise InputError('No compatible validated model is registered for: '+', '.join(unavailable)+'.')
                progress.progress(40,text='Loading selected validated model(s)…')
                with st.spinner('Combining MRI slices and imaging planes…'):
                    uncertainty_samples = cfg['evaluation'].get('uncertainty_samples', 1) if cfg['evaluation'].get('uncertainty_estimation', False) else 1
                    result=predictor.predict(
                        volumes, case or 'Not supplied', selection['selections'],
                        uncertainty_samples=uncertainty_samples
                    )
                selection['mode']='static_validation_routing'
                selection['dynamic_unavailable_reason']=dynamic_status['reason']
            result['input_intake']=intake;result['model_selection']=selection;result['ensemble_plan']=ensemble_plan;result['localization_status']=localizers
            st.session_state.result=result;st.session_state.volumes=volumes
            st.session_state.case_signature=current_signature;st.session_state.attention_cache={}
            progress.progress(100,text='Analysis complete')
        except (InputError,ValueError,FileNotFoundError) as e:
            if 'progress' in locals():progress.empty()
            error_msg = str(e)
            st.error(error_msg)
            
            # Provide specific guidance based on error type
            if 'upload limit' in error_msg.lower():
                st.info('💡 **Tip**: Compress your MRI files or reduce the number of slices. The current limit is 100 MB per study.')
            elif 'not a valid' in error_msg.lower() or 'could not be read' in error_msg.lower():
                st.info('💡 **Tip**: Ensure you are uploading valid .npy files or a properly structured DICOM ZIP. Check that files are not corrupted.')
            elif 'dimension' in error_msg.lower():
                st.info('💡 **Tip**: MRI stacks must be 3D arrays (slices × height × width) with dimensions between 16-1024 pixels and 1-512 slices.')
            elif 'confirm the mri plane' in error_msg.lower():
                st.info('💡 **Tip**: Choose the correct plane in the Study intake panel. The application will not guess orientation from an unlabelled processed array.')
            elif 'more than one' in error_msg.lower():
                st.info('💡 **Tip**: Upload only one sequence per plane. Remove duplicate files and try again.')
            elif 'at least one' in error_msg.lower():
                st.info('💡 **Tip**: Upload at least one MRI sequence (axial, coronal, or sagittal) to analyze the study.')
                
        except Exception:
            if 'progress' in locals():progress.empty()
            logging.exception('MRI analysis failed')
            st.error('The study could not be analysed due to an unexpected error.')
            st.info('💡 **Troubleshooting**: Check the file format and try again. If this continues, restart the application or check the application logs.')

with right:
    st.subheader('2 · Review research scores')
    if 'result' not in st.session_state:
        st.info('Add an MRI study and select Analyse study to view predictions and model attention.')
        st.markdown('**Three imaging planes, one study-level assessment**')
        st.caption('The model combines sampled slices across the available axial, coronal and sagittal sequences. An absent flag does not rule out injury.')
    else:
        result=st.session_state.result;volumes=st.session_state.volumes
        st.caption(f"Case: {result['case_reference']} · Model policy: {result['architecture']} · Sequences: {', '.join(result['available_planes'])}")
        routes = [
            f"{finding['finding']} → {display_architecture(finding.get('model_architecture', 'registered model'))}"
            for finding in result['findings']
        ]
        if result.get('dynamic_selection'):
            dynamic=result['dynamic_selection']
            st.info(
                '**Selected model for this study: '+display_architecture(dynamic['selected_architecture'])+'** · '
                'All five models analysed the same upload. The learned selector chose this architecture for all three targets using its lowest estimated mean Brier loss.'
            )
            st.caption(f"Models evaluated: {dynamic['models_evaluated']} of 5 · Estimated mean Brier loss: {dynamic['expected_losses'][dynamic['selected_model_id']]:.3f} (lower is preferred) · Analysis time: {dynamic['total_seconds']:.2f} s")
        else:
            st.info(
                '**Models used for this study:** ' + ' · '.join(routes) +
                '. Compatible routes were selected using saved validation evidence before case scoring.'
            )
        for start in range(0,len(result['findings']),3):
            group=result['findings'][start:start+3]
            for column,finding in zip(st.columns(len(group)),group):
                with column:
                    st.metric(finding['finding'],f"{finding['probability']:.1%}")
                    st.caption(f"Model used: {display_architecture(finding.get('model_architecture', 'registered model'))}")
                    st.caption(finding['status'])
                    # MC-dropout variation is displayed as an exploratory model-stability signal,
                    # never as clinical certainty.
                    if 'uncertainty' in finding:
                        st.caption(f"Exploratory score variation: {finding['uncertainty']:.3f} ({finding['uncertainty_level']}; not an injury severity measure)")
        st.caption(result['probability_note'])
        if result.get('routing_note'):
            st.info(result['routing_note'])
        if 'uncertainty_note' in result:
            st.info(result['uncertainty_note'])
        if result['missing_planes']:
            st.warning(
                'Input completeness caution - missing: '+', '.join(result['missing_planes'])+'. '
                'The uploaded study contains fewer sequences than the usual multi-plane MRI examination. '
                + result['incomplete_study_warning']
            )
        st.markdown(f"**Distance from research threshold: {result['score_separation']}** · Clinical confidence: not established")
        st.caption(result['confidence_note'])
        with st.expander('Model selection and study compatibility',expanded=True):
            selection=result.get('model_selection',{})
            st.caption('Available planes: '+', '.join(selection.get('available_planes',result['available_planes'])))
            if selection.get('mode') == 'dynamic_case_specific_competence_estimation':
                st.success(f"Dynamic selection completed: {selection['models_evaluated']} of 5 models evaluated. Selected architecture: {display_architecture(selection['selected_architecture'])}.")
                all_model_rows=[]
                for model_id, probabilities in selection['base_probabilities'].items():
                    all_model_rows.append({
                        'Model':display_architecture(model_id.replace('dms_', '')),
                        'General abnormality':f"{probabilities[0]:.1%}",
                        'ACL tear':f"{probabilities[1]:.1%}",
                        'Meniscal tear':f"{probabilities[2]:.1%}",
                        'Estimated mean Brier loss':f"{selection['expected_losses'][model_id]:.3f}",
                        'Selected':'Yes' if model_id == selection['selected_model_id'] else 'No',
                    })
                st.dataframe(pd.DataFrame(all_model_rows),hide_index=True,width='stretch')
                st.caption(selection['selection_rule'])
                st.caption(f"Selector estimator: {selection.get('selector_estimator', 'recorded artifact')}. Lower estimated loss is preferred; it is not a probability that the selection is correct.")
                if selection.get('principal_factors'):
                    st.markdown('**Principal factors in this selection estimate**')
                    factors = pd.DataFrame(selection['principal_factors']).copy()
                    if 'contribution_to_estimated_loss' in factors:
                        factors['contribution_to_estimated_loss'] = factors['contribution_to_estimated_loss'].round(4)
                        factors = factors.rename(columns={'contribution_to_estimated_loss': 'Contribution to estimated loss'})
                    elif 'feature_importance_mdi' in factors:
                        factors['feature_importance_mdi'] = factors['feature_importance_mdi'].round(4)
                        factors = factors.rename(columns={'feature_importance_mdi': 'Relative feature importance'})
                    st.dataframe(factors, hide_index=True, width='stretch')
                    st.caption(selection.get('factor_interpretation',''))
                if selection.get('selector_provenance_warnings'):
                    for warning in selection['selector_provenance_warnings']:
                        st.warning(warning)
                st.caption(selection['interpretation'])
            else:
                rows=[]
                for entry in selection.get('selections',[]):
                    metrics=entry.get('metrics',{})
                    rows.append({'Finding':entry['target'].replace('_',' ').title(),'Selected architecture':display_architecture(entry.get('architecture','Unavailable')),
                                 'Version':entry.get('version','—'),'Validation AUROC':f"{metrics.get('auroc',0):.3f}" if metrics else '—',
                                 'Sensitivity':f"{metrics.get('sensitivity',0):.3f}" if metrics else '—',
                                 'Average precision':f"{metrics.get('average_precision', metrics.get('pr_auc', 0)):.3f}" if metrics else '—'})
                if rows:st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
                st.caption(selection.get('policy',{}).get('rule','Registry evidence was used for selection.'))
                st.caption('Dynamic case-specific selection is unavailable: '+selection.get('dynamic_unavailable_reason','no verified dynamic selector artifact was found.'))
                st.caption('The displayed results use the existing saved validation-routing path. Individual prediction scores are not used to choose a route.')
            ensemble_plan={} if result.get('dynamic_selection') else result.get('ensemble_plan',{})
            for plan in ensemble_plan.get('plans',[]):
                if plan['status'] == 'Eligible validated ensemble':
                    st.caption(f"{plan['target'].replace('_',' ').title()}: validated ensemble {plan['ensemble_id']} is eligible.")
                else:
                    st.caption(f"{plan['target'].replace('_',' ').title()}: single-model fallback. {plan['reason']}")
        table_data=[{'Finding':f['finding'],'Model selected':f.get('model_architecture', result['architecture']),
                     'Model probability':f"{f['probability']:.1%}",
                     'Decision threshold':f"{f['threshold']:.2f}",'AI finding':f['status']}
                    for f in result['findings']]
        # Add uncertainty column if available
        if 'uncertainty' in result['findings'][0]:
            for i,f in enumerate(result['findings']):
                table_data[i]['MC-dropout variation'] = f"{f['uncertainty']:.3f} ({f['uncertainty_level']}; exploratory)"
        table=pd.DataFrame(table_data)
        st.dataframe(table,hide_index=True,width='stretch')
        with st.expander('Detection and segmentation status'):
            localizers=result.get('localization_status',{})
            for kind,label in [('detection','Detection boxes'),('segmentation','Segmentation masks')]:
                state=localizers.get(kind,{})
                if state.get('available'):
                    st.success(f'{label}: validated model available.')
                else:
                    st.info(f"{label}: unavailable. {state.get('reason','No approved model is registered.')}")
            st.caption('Grad-CAM remains classifier attention only; it is not a detection box or segmentation mask.')
        tabs=st.tabs(['MRI slice viewer','Model attention (not localisation)','Performance graphs','Export summary'])
        with tabs[0]:
            plane=st.selectbox('Viewer sequence',list(volumes),key='viewer_plane')
            v=volumes[plane];index=st.slider('MRI slice',0,len(v)-1,len(v)//2) if len(v)>1 else 0
            low,high=np.percentile(v,[1,99]);slice_image=np.clip((v[index].astype(float)-low)/max(high-low,1e-6),0,1)
            st.image(slice_image,caption=f'{plane.title()} · slice {index+1} of {len(v)}',width=380,clamp=True)
        with tabs[1]:
            targets={f['finding']:f['key'] for f in result['findings']}
            label=st.selectbox('Finding to explain',list(targets))
            plane=st.selectbox('Attention sequence',list(volumes),key='cam_plane')
            target_key=targets[label]
            key=target_key+'/'+plane
            finding=next((item for item in result['findings'] if item['key']==target_key),None)
            cached_attention=st.session_state.attention_cache.get(key)
            if not is_valid_attention_payload(cached_attention):
                cached_attention=None
            if cached_attention is None:
                st.info('Attention is generated only when requested. This keeps score review responsive and prevents a failed heatmap from interrupting study-level predictions.')
                generate_key='generate_attention_'+hashlib.sha256(key.encode()).hexdigest()[:12]
                if st.button('Generate attention map',key=generate_key):
                    try:
                        with st.spinner('Preparing model attention…'):
                            attention=get_predictor().explain(
                                volumes,target_key,plane,
                                model_id=finding.get('model_id') if finding else None,
                            )
                        if not is_valid_attention_payload(attention):
                            raise ValueError('The generated attention payload was incomplete. Please generate it again.')
                        st.session_state.attention_cache[key]=attention
                        result['explanation_status']='Generated for selected-model targets/sequences listed in explanations.'
                        result.setdefault('explanations',{})[key]={
                            'method':'Grad-CAM on final encoder spatial features',
                            'selected_model_id':finding.get('model_id') if finding else None,
                            'available':True,'has_positive_attention':attention['has_positive_attention'],
                            'sampled_original_slice_numbers':[int(i)+1 for i in attention['indices']],
                            'suggested_original_slice_number':int(attention['indices'][attention['suggested_slice']])+1,
                            'feature_map_size':list(attention['feature_map_size']),
                            'note':attention['note'],
                        }
                    except Exception:
                        logging.exception('Attention visualization failed')
                        st.error('Attention could not be generated for this view. The study predictions remain available.')
            else:
                st.caption('A current attention map is available for this finding and sequence.')

            attention=st.session_state.attention_cache.get(key)
            if is_valid_attention_payload(attention):
                try:
                    st.subheader(f'{plane.title()} plane score explanation')
                    st.caption('One sequence and one sampled slice are shown at a time. Cross-plane averaging is intentionally unavailable because an averaged overlay is not an anatomical view.')
                    idx=st.select_slider('Sampled MRI slice',options=list(range(len(attention['indices']))),value=attention['suggested_slice'],
                                         format_func=lambda i:f"Slice {attention['indices'][i]+1}",key='cam_slice_'+key)

                    a,b=st.columns(2)
                    a.image(attention['original'][idx],caption='Original MRI slice',width='stretch',clamp=True)
                    b.image(attention['overlay'][idx],caption='Grad-CAM positive contribution overlay',width='stretch',clamp=True)

                    probability=float(finding['probability']) if finding else None
                    region=attention['regions'][idx]
                    st.markdown('#### What this attention view means')
                    probability_text=f" The model score for this finding is {probability:.1%}." if probability is not None else ''
                    st.info(
                        f"This overlay explains the model's **{label}** score on the **{plane.title()}** sequence, "
                        f"shown at **slice {attention['indices'][idx]+1}**.{probability_text} "
                        "Warmer pixels show image areas that contributed more positively to this selected score; they do not show a tear."
                    )
                    if region.get('available'):
                        st.markdown(
                            f"**Highest contribution area in this display:** {region['display_region'].capitalize()}. "
                            f"The top part of this heatmap covers about {region['highlighted_area_percent']:.1f}% of the displayed image."
                        )
                        st.caption(
                            f"Peak attention pixel: row {region['peak_pixel']['row']}, column {region['peak_pixel']['column']} "
                            f"in this resized display. The displayed map was enlarged from a {attention.get('feature_map_size',('unknown','unknown'))[0]} × "
                            f"{attention.get('feature_map_size',('unknown','unknown'))[1]} feature map, so it is coarse. These are image coordinates, not a named knee structure."
                        )
                    else:
                        st.info(region.get('reason','No concentrated Grad-CAM region is available for this slice.'))
                    st.warning(
                        "Interpretation boundary: this is a coarse classifier explanation only. It does not confirm a tear, "
                        "identify the exact tissue involved, provide a lesion boundary, estimate lesion size, or exclude pathology elsewhere in the study."
                    )
                    st.caption(attention['note'])
                    if not attention['has_positive_attention']:
                        st.info('No positive Grad-CAM signal for this finding in this sequence. No region is highlighted.')

                    if st.button('Export attention heatmap'):
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(8, 8))
                        ax.imshow(attention['overlay'][idx])
                        ax.axis('off')
                        ax.set_title(f"{label} - {plane} - Slice {attention['indices'][idx]+1}")
                        plt.tight_layout()
                        import io
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                        buf.seek(0)
                        st.download_button('Download attention heatmap', buf.getvalue(),
                                         file_name=f'attention_{label}_{plane}_slice{attention["indices"][idx]+1}.png',
                                         mime='image/png')
                        plt.close(fig)
                except Exception:
                    logging.exception('Attention visualization failed')
                    st.error('Attention could not be displayed for this view. The study predictions remain available.')
        with tabs[2]:
            st.subheader('Model Performance Graphs')
            st.caption('Saved cohort evaluation of the analysis pipeline. These graphs are not a single uploaded patient result.')

            routed_results_dir = (ROOT / 'runs' / 'dynamic_selection_v1' / 'figures') if result.get('dynamic_selection') else (ROOT / 'runs' / 'model_portfolio_v1' / 'final')
            final_results_dir = routed_results_dir
            if routed_results_dir.exists():
                st.caption('These results use the previously exposed MRNet official-validation cohort, held out from this run fitting. They are not independent clinical validation.')
            
            # Check if performance graphs exist
            roc_curves_path = final_results_dir / 'roc_curves.png'
            pr_curves_path = final_results_dir / 'precision_recall_curves.png'
            confusion_matrix_path = final_results_dir / 'confusion_matrices.png'
            metrics_path = final_results_dir / 'metrics.json'
            
            if roc_curves_path.exists():
                st.image(str(roc_curves_path), caption='ROC Curves - Internal Validation Performance', width='stretch')
            else:
                st.info('ROC curves not available. Run full evaluation to generate performance graphs.')
            
            if pr_curves_path.exists():
                st.image(str(pr_curves_path), caption='Precision-Recall Curves - Internal Validation Performance', width='stretch')
            
            if confusion_matrix_path.exists():
                st.image(str(confusion_matrix_path), caption='Confusion Matrices - Per-Finding Performance', width='stretch')
            
            # Display numerical metrics if available
            if metrics_path.exists():
                import json
                with open(metrics_path, 'r') as f:
                    metrics_data = json.load(f)
                metric_findings = metrics_data.get('metrics', metrics_data)
                
                st.subheader('Detailed Performance Metrics')
                metrics_df_data = []
                for finding_key, finding_metrics in metric_findings.items():
                    if finding_key != 'macro' and isinstance(finding_metrics, dict):
                        metrics_df_data.append({
                            'Finding': finding_key.replace('_', ' ').title(),
                            'AUROC': f"{finding_metrics.get('auroc', 0):.3f}",
                            'F1 Score': f"{finding_metrics.get('f1', 0):.3f}",
                            'Sensitivity': f"{finding_metrics.get('sensitivity', 0):.3f}",
                            'Specificity': f"{finding_metrics.get('specificity', 0):.3f}",
                            'Precision': f"{finding_metrics.get('precision', 0):.3f}",
                            'Accuracy': f"{finding_metrics.get('accuracy', 0):.3f}"
                        })
                
                if metrics_df_data:
                    import pandas as pd
                    metrics_df = pd.DataFrame(metrics_df_data)
                    st.dataframe(metrics_df, hide_index=True, width='stretch')
            
            st.info('Performance graphs and metrics are saved development evidence. They are not independent clinical validation results.')
        with tabs[3]:
            summary=text_summary(result)
            st.text_area('AI results summary',summary,height=300)
            a,b=st.columns(2)
            a.download_button('Download text summary',summary,file_name='KneeAssist_AI_summary.txt',mime='text/plain',width='stretch')
            b.download_button('Download structured JSON',json.dumps(result,indent=2),file_name='KneeAssist_AI_summary.json',mime='application/json',width='stretch')
st.divider()
st.caption('KneeAssist XAI V1 · Explainable research prototype trained on MRNet · Not for autonomous diagnosis · Clinical review required')
