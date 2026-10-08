# Usability evaluation plan

## Purpose

Evaluate whether the KneeAssist XAI research prototype can be used to upload one knee MRI study, understand the displayed research output, view model attention and export a summary. This is an interface usability evaluation, not a clinical diagnostic validation.

## Participants

Recruit 5-10 adult volunteer users such as final-year computing students, researchers or appropriately supervised healthcare users. Do not upload patient-identifiable data. Record participant codes only, for example `U01`.

## Tasks

1. Start the local dashboard and load the packaged research example.
2. Identify the three displayed finding probabilities.
3. Open the MRI slice viewer and change the displayed sequence or slice.
4. Open the model-attention tab and state what Grad-CAM does and does not mean.
5. Export the research summary.
6. Clear the current case.
7. Attempt an invalid file upload and confirm that the error message is understandable.

## Measures

- Task completion: completed, completed with help, or not completed.
- Time in seconds for each task.
- Number and description of user errors.
- Five-point agreement rating for the statements in `usability_responses_template.csv`.
- One free-text improvement suggestion.

## Procedure

Give each participant the same task sheet. Do not coach them except when a task cannot be completed. Record observations without personal health information. Calculate task-completion percentage, median task time, average rating per question and common themes in comments.

## Interpretation boundary

A usability result shows whether the prototype interface is understandable. It does not show diagnostic accuracy, clinician agreement, patient benefit, safety or clinical readiness.
