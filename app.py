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
from src.selection.ensemble import ensemble_plans
from src.localization.registry import localization_status
from src.inference.predictor import Predictor,text_summary

st.set_page_config(page_title='KneeAssist AI · MRI review',page_icon='🩻',layout='wide')
cfg=config()
if not cfg.get('lifecycle',{}).get('inference_enabled',True):
    st.title('KneeAssist AI')
    st.info('MRI analysis is disabled in the current deployment configuration.')
    st.caption('See START_HERE.md for the active model and verification status.')
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

@st.cache_resource
def cached_predictor(checkpoint_stamp):
    return Predictor(cfg=cfg)

def get_predictor():
    checkpoint=resolve(cfg['evaluation']['checkpoint'])
    stamp=(str(checkpoint),checkpoint.stat().st_mtime_ns,checkpoint.stat().st_size) if checkpoint.exists() else ('unavailable',)
    return cached_predictor(stamp)

if 'upload_generation' not in st.session_state:st.session_state.upload_generation=0

def clear_case():
    generation=st.session_state.upload_generation+1
    st.session_state.clear()
    st.session_state.upload_generation=generation

with st.sidebar:
    st.markdown('## KneeAssist AI')
    st.caption('MRI RESEARCH WORKSPACE · V1')
    st.divider()
    st.markdown('**Study-level review**\n\n'+'\n\n'.join(t['display'] for t in cfg['targets']))
    st.divider()
    st.caption('Runs locally. Uploaded images and case references remain in this browser session and are not written to application logs.')
    st.button('Clear current case',use_container_width=True,on_click=clear_case)
    st.caption('Research and clinical decision support. Not an autonomous diagnostic system.')

st.markdown('<div class="intro"><div class="tag">Knee MRI · research review</div><h1>KneeAssist AI</h1><p>Review study-level model scores and attention maps. Clinical judgment remains essential.</p></div>',unsafe_allow_html=True)
st.warning('Research decision-support prototype. Clinical review is required. Model attention is not a confirmed lesion or lesion location.')
with st.expander('Model evidence and supported inputs'):
    from src.inference.readiness import deployment_evidence
    evidence=deployment_evidence(cfg)
    st.caption('Active checkpoint: '+evidence['checkpoint'])
    st.markdown('**Use one knee MRI study at a time.** All uploaded sequences must belong to the same case. File checks cannot confirm anatomy, patient identity or acquisition suitability.')
    st.caption('MRI NumPy stacks, NIfTI volumes and DICOM studies are inspected before analysis. Unlabelled processed arrays require manual plane confirmation. JPEG/PNG are not supported. No finding rules out an injury.')
    if evidence['metrics']:
        st.caption('Previously evaluated MRNet validation cohort. These measurements are not clinical acceptance criteria.')
        st.dataframe(pd.DataFrame([{'Finding':t['display'],'AUROC':round(evidence['metrics'][t['key']]['auroc'],3),
                       'Sensitivity':round(evidence['metrics'][t['key']]['sensitivity'],3),
                       'Specificity':round(evidence['metrics'][t['key']]['specificity'],3)} for t in cfg['targets']]),hide_index=True)
    st.caption('Training uses MRNet only. External KneeMRI was previously evaluated; fastMRI annotations provide limited reference labels. Patient independence and clinical confidence are not established.')
left,right=st.columns([1.05,1.8],gap='large')
with left:
    st.subheader('1 · Add a study')
    case=st.text_input('Case reference',placeholder='e.g. RESEARCH-001',max_chars=120,key=f'case_reference_{st.session_state.upload_generation}')
    uploads=st.file_uploader('MRI study files',type=['npy','zip','dcm','nii','gz'],accept_multiple_files=True,
        help='Use .npy, .nii/.nii.gz, .dcm, or ZIP study files. The intake panel inspects the upload before prediction.',
        key=f'uploads_{st.session_state.upload_generation}')
    st.caption('Supported: MRI NumPy stacks, NIfTI volumes, DICOM slices and study ZIPs. File checks cannot confirm anatomy, mixed patients or clinical suitability. JPEG/PNG are rejected.')
    if st.button('Load research example',use_container_width=True):
        demo=ROOT/'sample_cases/mrnet_1130.zip'
        if uploads:
            st.info('Clear the current case before loading the packaged example.')
        elif demo.exists():
            st.session_state.demo=('mrnet_1130.zip',demo.read_bytes());st.session_state.pop('result',None)
        else:st.error('The packaged example is unavailable. Upload a study instead.')
    if uploads:st.session_state.pop('demo',None)
    elif 'demo' in st.session_state:st.info('Research example loaded: MRNet validation study 1130.')
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
                st.dataframe(pd.DataFrame(intake_rows),hide_index=True,use_container_width=True)
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
    active_checkpoint=resolve(cfg['evaluation']['checkpoint'])
    model_stamp=(str(active_checkpoint),active_checkpoint.stat().st_mtime_ns,active_checkpoint.stat().st_size) if active_checkpoint.exists() else ('unavailable',)
    current_signature=hashlib.sha256(b''.join(n.encode()+b for n,b in files)+case.encode()+json.dumps(plane_overrides,sort_keys=True).encode()+repr(model_stamp).encode()).hexdigest()
    # Never show a previous case's predictions against changed inputs.
    if st.session_state.get('case_signature')!=current_signature:
        for key in ['result','volumes','attention_cache']:st.session_state.pop(key,None)
        for key in list(st.session_state):
            if key.startswith('cam_slice_'):st.session_state.pop(key,None)
    if st.button('Analyse study',type='primary',use_container_width=True):
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
            volumes=load_uploads(files,cfg['preprocessing'],plane_overrides);progress.progress(25,text='Selecting compatible validated model…')
            selection=select_models(list(volumes),[target['key'] for target in cfg['targets']])
            ensemble_plan=ensemble_plans(list(volumes),[target['key'] for target in cfg['targets']])
            localizers=localization_status()
            unavailable=[entry['target'] for entry in selection['selections'] if entry['status']!='Selected']
            if unavailable:
                raise InputError('No compatible validated model is registered for: '+', '.join(unavailable)+'.')
            selected_paths={entry['checkpoint'] for entry in selection['selections']}
            if selected_paths!={cfg['evaluation']['checkpoint']}:
                raise InputError('The selected registry checkpoint does not match the active inference checkpoint. Update the registry before analysis.')
            progress.progress(40,text='Loading trained model…')
            predictor=get_predictor();progress.progress(50,text='Analysing MRI sequences…')
            with st.spinner('Combining MRI slices and imaging planes…'):
                # Enable uncertainty quantification with Monte Carlo dropout if configured
                uncertainty_samples = cfg['evaluation'].get('uncertainty_samples', 1) if cfg['evaluation'].get('uncertainty_estimation', False) else 1
                result=predictor.predict(volumes,case or 'Research case',uncertainty_samples=uncertainty_samples)
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
        st.caption(f"Case: {result['case_reference']} · Model: {result['architecture']} · Sequences: {', '.join(result['available_planes'])}")
        for start in range(0,len(result['findings']),3):
            group=result['findings'][start:start+3]
            for column,finding in zip(st.columns(len(group)),group):
                with column:
                    st.metric(finding['finding'],f"{finding['probability']:.1%}")
                    st.caption(finding['status'])
                    # MC-dropout variation is displayed as an exploratory model-stability signal,
                    # never as clinical certainty.
                    if 'uncertainty' in finding:
                        st.caption(f"MC-dropout variation: {finding['uncertainty']:.3f} ({finding['uncertainty_level']}; exploratory)")
        st.caption(result['probability_note'])
        if 'uncertainty_note' in result:
            st.info(result['uncertainty_note'])
        if result['missing_planes']:st.warning('Missing: '+', '.join(result['missing_planes'])+'. '+result['incomplete_study_warning'])
        st.markdown(f"**Distance from research threshold: {result['score_separation']}** · Clinical confidence: not established")
        st.caption(result['confidence_note'])
        with st.expander('Validated model selection and study compatibility',expanded=True):
            selection=result.get('model_selection',{})
            st.caption('Available planes: '+', '.join(selection.get('available_planes',result['available_planes'])))
            rows=[]
            for entry in selection.get('selections',[]):
                metrics=entry.get('metrics',{})
                rows.append({'Finding':entry['target'].replace('_',' ').title(),'Selected architecture':entry.get('architecture','Unavailable'),
                             'Version':entry.get('version','—'),'Validation AUROC':f"{metrics.get('auroc',0):.3f}" if metrics else '—',
                             'Sensitivity':f"{metrics.get('sensitivity',0):.3f}" if metrics else '—',
                             'PR-AUC':f"{metrics.get('pr_auc',0):.3f}" if metrics else '—'})
            if rows:st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
            st.caption(selection.get('policy',{}).get('rule','Registry evidence was used for selection.'))
            ensemble_plan=result.get('ensemble_plan',{})
            for plan in ensemble_plan.get('plans',[]):
                if plan['status'] == 'Eligible validated ensemble':
                    st.caption(f"{plan['target'].replace('_',' ').title()}: validated ensemble {plan['ensemble_id']} is eligible.")
                else:
                    st.caption(f"{plan['target'].replace('_',' ').title()}: single-model fallback. {plan['reason']}")
        table_data=[{'Finding':f['finding'],'Model probability':f"{f['probability']:.1%}",
                     'Decision threshold':f"{f['threshold']:.2f}",'AI finding':f['status']}
                    for f in result['findings']]
        # Add uncertainty column if available
        if 'uncertainty' in result['findings'][0]:
            for i,f in enumerate(result['findings']):
                table_data[i]['MC-dropout variation'] = f"{f['uncertainty']:.3f} ({f['uncertainty_level']}; exploratory)"
        table=pd.DataFrame(table_data)
        st.dataframe(table,hide_index=True,use_container_width=True)
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
            label=st.selectbox('Finding to explain',list(targets));plane=st.selectbox('Attention sequence',list(volumes),key='cam_plane')
            key=targets[label]+'/'+plane
            try:
                cached_attention=st.session_state.attention_cache.get(key)
                expected_slices=len(cached_attention.get('indices',[])) if isinstance(cached_attention,dict) else 0
                cache_is_current=(
                    isinstance(cached_attention,dict)
                    and cached_attention.get('schema_version')==2
                    and isinstance(cached_attention.get('regions'),list)
                    and len(cached_attention['regions'])==expected_slices
                )
                if not cache_is_current:
                    with st.spinner('Preparing model attention…'):
                        st.session_state.attention_cache[key]=get_predictor().explain(volumes,targets[label],plane)
                attention=st.session_state.attention_cache[key]
                st.subheader(f'{plane.title()} plane score explanation')
                st.caption('One sequence and one sampled slice are shown at a time. Cross-plane averaging is intentionally unavailable because an averaged overlay is not an anatomical view.')
                idx=st.select_slider('Sampled MRI slice',options=list(range(len(attention['indices']))),value=attention['suggested_slice'],
                                     format_func=lambda i:f"Slice {attention['indices'][i]+1}",key='cam_slice_'+key)
                
                a,b=st.columns(2)
                a.image(attention['original'][idx],caption='Original MRI slice',use_container_width=True,clamp=True)
                b.image(attention['overlay'][idx],caption='Grad-CAM positive contribution overlay',use_container_width=True,clamp=True)

                finding=next((item for item in result['findings'] if item['key']==targets[label]),None)
                probability=float(finding['probability']) if finding else None
                regions=attention.get('regions') or []
                region=regions[idx] if idx<len(regions) else {
                    'available':False,
                    'reason':'This attention view has no region summary. Please select the finding or sequence again to regenerate it.'
                }
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
                if not attention['has_positive_attention']:st.info('No positive Grad-CAM signal for this finding in this sequence. No region is highlighted.')
                
                # Export attention heatmap
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
                st.error('Attention could not be generated for this view. The study predictions remain available.')
        with tabs[2]:
            st.subheader('Model Performance Graphs')
            st.caption('Performance metrics from internal validation and external evaluation datasets.')
            
            results_dir = resolve(cfg['evaluation']['directory'])
            final_results_dir = results_dir / 'final'
            
            # Check if performance graphs exist
            roc_curves_path = final_results_dir / 'roc_curves.png'
            pr_curves_path = final_results_dir / 'precision_recall_curves.png'
            confusion_matrix_path = final_results_dir / 'confusion_matrices.png'
            metrics_path = final_results_dir / 'metrics.json'
            
            if roc_curves_path.exists():
                st.image(str(roc_curves_path), caption='ROC Curves - Internal Validation Performance', use_container_width=True)
            else:
                st.info('ROC curves not available. Run full evaluation to generate performance graphs.')
            
            if pr_curves_path.exists():
                st.image(str(pr_curves_path), caption='Precision-Recall Curves - Internal Validation Performance', use_container_width=True)
            
            if confusion_matrix_path.exists():
                st.image(str(confusion_matrix_path), caption='Confusion Matrices - Per-Finding Performance', use_container_width=True)
            
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
                    st.dataframe(metrics_df, hide_index=True, use_container_width=True)
            
            # External evaluation results if available
            external_results = {}
            for external_name in ['external_kneemri', 'external_fastmri']:
                external_dir = results_dir / external_name
                if external_dir.exists():
                    external_json = external_dir / 'metrics.json'
                    if external_json.exists():
                        with open(external_json, 'r') as f:
                            external_results[external_name] = json.load(f)
            
            if external_results:
                st.subheader('External Dataset Evaluation')
                for external_name, external_data in external_results.items():
                    st.markdown(f"**{external_name.replace('_', ' ').title()}**")
                    external_df_data = []
                    external_metrics = external_data.get('metrics', external_data.get('primary', external_data))
                    for finding_key, finding_metrics in external_metrics.items():
                        if finding_key != 'macro' and isinstance(finding_metrics, dict):
                            external_df_data.append({
                                'Finding': finding_key.replace('_', ' ').title(),
                                'AUROC': f"{finding_metrics.get('auroc', 0):.3f}",
                                'Sensitivity': f"{finding_metrics.get('sensitivity', 0):.3f}",
                                'Specificity': f"{finding_metrics.get('specificity', 0):.3f}"
                            })
                    
                    if external_df_data:
                        external_df = pd.DataFrame(external_df_data)
                        st.dataframe(external_df, hide_index=True, use_container_width=True)
            
            # Model comparison if available
            model_comparison_path = results_dir / 'model_comparison.json'
            if model_comparison_path.exists():
                with open(model_comparison_path, 'r') as f:
                    comparison_data = json.load(f)
                
                st.subheader('Model Comparison')
                st.markdown(f"**Selected Architecture**: {comparison_data.get('winner', 'N/A')}")
                st.caption(f"Selection Criterion: {comparison_data.get('criterion', 'N/A')}")
                
                if 'models' in comparison_data:
                    comparison_df_data = []
                    for model_info in comparison_data['models']:
                        comparison_df_data.append({
                            'Architecture': model_info.get('architecture', 'N/A'),
                            'Stage': model_info.get('stage', model_info.get('candidate', 'N/A')),
                            'Best Epoch': model_info.get('selected_epoch', 'N/A'),
                            'Internal AUROC': f"{model_info.get('metrics', {}).get('macro', {}).get('auroc', 0):.4f}"
                        })
                    
                    if comparison_df_data:
                        comparison_df = pd.DataFrame(comparison_df_data)
                        st.dataframe(comparison_df, hide_index=True, use_container_width=True)
            
            st.info('Performance graphs and metrics are from the model evaluation phase. They represent internal validation performance and external reference dataset evaluation. These are not clinical validation results.')
        with tabs[3]:
            summary=text_summary(result)
            st.text_area('AI results summary',summary,height=300)
            a,b=st.columns(2)
            a.download_button('Download text summary',summary,file_name='KneeAssist_AI_summary.txt',mime='text/plain',use_container_width=True)
            b.download_button('Download structured JSON',json.dumps(result,indent=2),file_name='KneeAssist_AI_summary.json',mime='application/json',use_container_width=True)
st.divider()
st.caption('KneeAssist AI V1 · Research prototype trained on MRNet · Not for autonomous diagnosis · Clinical review required')
