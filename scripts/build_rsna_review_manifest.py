"""Create a de-identified review manifest from mounted RSNA DICOM data.

Run this in Kaggle after attaching the RSNA competition input. It reads only
metadata, does not copy MRI pixels, and never exports patient names or IDs.
"""
import argparse
import csv
import hashlib
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pydicom


TAGS = [
    'StudyInstanceUID', 'SeriesInstanceUID', 'Modality', 'ImageOrientationPatient',
    'Manufacturer', 'ManufacturerModelName', 'MagneticFieldStrength',
]


def plane(dataset):
    """Return a conservative acquisition-plane name from direction cosines."""
    try:
        orientation = np.asarray(dataset.ImageOrientationPatient, dtype=float)
        if orientation.shape != (6,):
            return 'unknown'
        axis = int(np.argmax(np.abs(np.cross(orientation[:3], orientation[3:]))))
        return ('sagittal', 'coronal', 'axial')[axis]
    except (AttributeError, TypeError, ValueError):
        return 'unknown'


def pseudonym(study_uid, salt):
    return 'RSNA-' + hashlib.sha256(f'{salt}:{study_uid}'.encode()).hexdigest()[:16].upper()


def read_header(path):
    try:
        return pydicom.dcmread(path, stop_before_pixels=True, specific_tags=TAGS, force=False)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description='Build a de-identified RSNA label-review manifest.')
    parser.add_argument('--dicom-root', type=Path, required=True, help='Mounted Kaggle DICOM input directory')
    parser.add_argument('--output', type=Path, required=True, help='CSV manifest to create')
    parser.add_argument('--salt-env', default='RSNA_REVIEW_SALT', help='Environment variable holding a private pseudonymisation salt')
    args = parser.parse_args()
    salt = os.environ.get(args.salt_env)
    if not salt:
        raise SystemExit(f'Set {args.salt_env} before running. Do not commit its value.')
    if not args.dicom_root.is_dir():
        raise SystemExit('The supplied DICOM root does not exist.')

    studies = defaultdict(lambda: {'series': defaultdict(dict), 'instances': 0})
    candidates = [p for p in args.dicom_root.rglob('*') if p.is_file() and (p.suffix.lower() == '.dcm' or not p.suffix)]
    for path in candidates:
        dataset = read_header(path)
        if dataset is None:
            continue
        study_uid = str(getattr(dataset, 'StudyInstanceUID', ''))
        series_uid = str(getattr(dataset, 'SeriesInstanceUID', ''))
        if not study_uid or not series_uid or str(getattr(dataset, 'Modality', '')) != 'MR':
            continue
        series = studies[study_uid]['series'][series_uid]
        series['plane'] = plane(dataset)
        series['manufacturer'] = str(getattr(dataset, 'Manufacturer', 'unknown'))[:80]
        series['model'] = str(getattr(dataset, 'ManufacturerModelName', 'unknown'))[:80]
        series['field_strength'] = str(getattr(dataset, 'MagneticFieldStrength', 'unknown'))[:20]
        studies[study_uid]['instances'] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as handle:
        fields = ['case_id', 'series_count', 'instance_count', 'available_planes', 'manufacturer_groups', 'field_strengths', 'review_status']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for study_uid, record in sorted(studies.items()):
            series = list(record['series'].values())
            writer.writerow({
                'case_id': pseudonym(study_uid, salt),
                'series_count': len(series),
                'instance_count': record['instances'],
                'available_planes': ';'.join(sorted({item['plane'] for item in series})),
                'manufacturer_groups': ';'.join(sorted({item['manufacturer'] for item in series})),
                'field_strengths': ';'.join(sorted({item['field_strength'] for item in series})),
                'review_status': 'pending',
            })
    print(f'Created {args.output} with {len(studies)} de-identified MRI studies.')


if __name__ == '__main__':
    main()
