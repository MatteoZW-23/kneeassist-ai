"""Validate two independent review rounds and create an adjudication queue.

This script never infers a diagnosis. A case becomes training-eligible only
when two reviewers agree on all three configured targets and image quality is
marked acceptable by both reviewers.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path


TARGETS = ('general_abnormality', 'acl_tear', 'meniscal_tear')
REQUIRED = {'case_id', 'reviewer_id', 'image_quality', *TARGETS}
VALID_LABELS = {'0', '1', 'uncertain'}


def rows(path):
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'Missing required columns: {", ".join(sorted(missing))}')
        return list(reader)


def write(path, fields, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description='Create agreed labels and an adjudication queue.')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True, help='Combined reviewer CSV using the supplied template')
    parser.add_argument('--approved-output', type=Path, required=True)
    parser.add_argument('--queue-output', type=Path, required=True)
    args = parser.parse_args()

    with args.manifest.open(newline='', encoding='utf-8') as handle:
        known_cases = {row['case_id'] for row in csv.DictReader(handle) if row.get('case_id')}
    reviews = defaultdict(list)
    for row in rows(args.labels):
        case_id = row['case_id'].strip()
        if case_id not in known_cases:
            raise ValueError(f'Unknown case_id: {case_id}')
        if not row['reviewer_id'].strip():
            raise ValueError(f'Missing reviewer_id for {case_id}')
        if row['image_quality'].strip().lower() not in {'acceptable', 'unacceptable', 'uncertain'}:
            raise ValueError(f'Invalid image_quality for {case_id}')
        for target in TARGETS:
            if row[target].strip().lower() not in VALID_LABELS:
                raise ValueError(f'Invalid {target} label for {case_id}')
        reviews[case_id].append(row)

    approved, queue = [], []
    for case_id in sorted(known_cases):
        case_reviews = reviews.get(case_id, [])
        reviewer_ids = {row['reviewer_id'].strip() for row in case_reviews}
        reason = None
        if len(reviewer_ids) != 2:
            reason = 'requires_two_distinct_reviewers'
        elif any(row['image_quality'].strip().lower() != 'acceptable' for row in case_reviews):
            reason = 'image_quality_not_acceptable'
        elif any(row[target].strip().lower() == 'uncertain' for row in case_reviews for target in TARGETS):
            reason = 'uncertain_label'
        elif any(len({row[target].strip().lower() for row in case_reviews}) != 1 for target in TARGETS):
            reason = 'reviewer_disagreement'
        if reason:
            queue.append({'case_id': case_id, 'adjudication_reason': reason})
        else:
            approved.append({'case_id': case_id, **{target: case_reviews[0][target].strip().lower() for target in TARGETS}, 'label_source': 'two_reviewer_agreement'})

    write(args.approved_output, ['case_id', *TARGETS, 'label_source'], approved)
    write(args.queue_output, ['case_id', 'adjudication_reason'], queue)
    print(f'Approved {len(approved)} cases; queued {len(queue)} cases for adjudication.')


if __name__ == '__main__':
    main()
