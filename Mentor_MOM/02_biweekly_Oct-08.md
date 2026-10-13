# Minutes - Biweekly Meeting with Mentor

**Date:** 8 October 2026

**Time:** 16:00 - 17:00 CEST

**Attendees:** 
- Mentor - Prof. Dr. Magda Gregorová

- Team - Pavin Sumathi Palanichamy, Prethebha Muthukumaran, Priyam Jha.

- Guest - Amparo Zamora Mogollo -PhD Student.

---

## Project status and context

The team presented the current state of CellFoundry, the collaboration with IMES (Institut für Medizintechnik Schweinfurt, THWS), for the guest and the mentor. The core task is instance segmentation of adipocytes (fat cells) in microscopy images provided by IMES.

As understood by the mentor, neither dataset is real tissue. Both are in-vitro cultures: cells are extracted from humans and pigs (bone marrow) and induced to differentiate into fat cells. This understanding has not been confirmed with IMES.
- **Porcine (animal) cells** - application is lab-grown meat, where fat cells are needed for flavor.
- **Human cells** - application is obesity and type 2 diabetes research, for example by counting fat cells and measuring their size.

The cultures contain adipocytes and stem cells. Fat cells can be recognized by the small dots inside them, which are likely lipids. In the porcine images the final fat cells are rounder and more symmetric, while the human cells are more elongated, with overlapping and diffuse boundaries. This is why human cells are hard to annotate, even for IMES's own annotators.

The masks contain no information about cell type, so the adipocyte vs. stem cell distinction would have to come from visual features and from the model itself. The team treats this as a secondary task.

## Priorities and feedback received from IMES

IMES replied by email on the day of the meeting:
- **Porcine bright-field images are the priority.** Phase-contrast images are not the priority.
- **Human cells** are lower priority but would be appreciated if segmentation is achievable.
- **Any model may be used**, on the condition that the results can be integrated into IMES's app.
- **Cellpose-SAM performs worse than Dinocell**, so the team does not need to reproduce the Cellpose-SAM results. IMES is working with Dinocell, which is now the only baseline.
- **Evaluation metrics, training curves and image overlays** were shared and confirmed to be generic metrics (precision, recall, F1).
- **Full code and settings** will follow by the next Monday (12 October 2026). The Colab notebook supports only FP16 precision, while IMES's own app supports BFloat16, and the preprocessing code sent so far is incomplete. The team decided not to study the incomplete preprocessing code, to avoid being misled by it.
- The results should have practical benefit for the users.

## Data received

- **Porcine bright-field images** - priority data, confirmed by IMES.
- **36 additional unannotated human bright-field images** - to be used for pseudo-labelling or as test images once a model exists.
- **105 additional unannotated porcine bright-field images** - extra data for the priority group.

**Format discrepancies:** the microscope images are PNG, while masks are single-channel instance label images in TIFF. Some folders also hold annotations as PNG, or as a mix of TIF, TIFF and PNG. A clear standardization step is needed. The team has written a script to visualize the images and masks.

**Time dimension in the data:** the porcine phase-contrast filenames carry the day of culture. Labeled cell counts rise sharply over the days, for example 2 cells at day 0 and 325 at day 14, and the fat cells grow into tight clusters that are hard to segment.

## Mentor feedback

1. **Clarify why segmentation is needed.** If only the proportion of fat is wanted, simple color or area calculations would be enough and no segmentation is required. The team argued that segmentation is needed for cell count and for cell size (hypertrophy: larger cells release more fatty acids, which reduces insulin sensitivity). The mentor asked the team to discuss this with Nicole, since IMES may not know what the project could offer or what it really needs.

2. **Adjust the scope.** Drop the porcine phase-contrast step. Expand the porcine bright-field step into a proper comparison of more than one model, which is more valuable than extending to phase contrast, a different domain to which models will not transfer easily. Then finish with the counts and morphology on bright-field images, so that a complete pipeline exists that others can later extend to phase contrast.

3. **Choice of models.** Any model may be used. The mentor's impression is that language-prompted models raise high expectations but give modest results, because the outcome depends heavily on the prompt. The team noted that SAM prompts are not tuned for microscopy or medical data.

4. **Preprocessing.** Some preprocessing is believed to be needed. The team expects contrast enhancement to help. The preprocessing steps proposed in the project brief should be checked for whether they make sense.

5. **Metrics.** No human study will be done, so the team sticks to computed metrics. Threshold-dependent metrics (F1 at 50% vs. 70% overlap) cannot be reduced to one target, so both should be reported. Metric preferences should not be left open for the client to decide: the team should work out the pros and cons, give a recommendation, and say what is achievable. The aim is to beat IMES's baseline.

6. **Experiment tracking (high priority).** Stop printing results to PDFs and copying values back by hand. Build an experimental protocol and a tracking system (for example Weights & Biases) that records parameters, preprocessing, augmentations, results and plots. With several models, preprocessing variants and augmentations, the mentor estimated easily 150 experiments, and typically 300 to 500 configurations. Run only the models the team has agreed on, and run things cleanly instead of in bulk.

7. **Limited annotated data.** There are only about 30 annotated images. Augmentation (beyond simple rotations) combined with fine-tuning, as discussed last semester, are the main options. Try the models out of the box first, then preprocessing, and only then fine-tuning and augmentations. Ask Nicole how much annotated data exists. More unannotated data is available, but it does not help directly.

8. **Distribute the work.** Roles have not been assigned yet. The mentor advised that the work needs to be split: someone reproduces IMES's results, someone reads up on candidate models, and someone sets up the experiment tracking.

## Evaluation and next steps, as planned by the team

1. Understand the data, preprocess the images and settle the evaluation metrics once IMES's code arrives.
2. Reproduce IMES's Dinocell baseline, then confirm with Nicole that everything meets the requirements, not only the numbers. Reproducing and establishing the baseline are merged into one step: the model weights and metrics are frozen as the reference.
3. Benchmark several candidate models (at least three) on the porcine bright-field images.
4. Compute cell count, cell size and morphology, including the width-to-length ratio of each cell.
5. Human cells and pseudo-labelling of the unannotated images, as additional improvements.

## Team workflow

- **GitHub:** a CellFoundry organization and repository. Meeting minutes (client and mentor meetings), README files of the data and shared files are kept there, but not the data itself, which is confidential and exceeds GitHub's 100 MB limit.
- **Branches:** everyone works on a separate branch. Merging into main requires review by the other two team members.
- **Milestones and issues:** tracked with GitHub Projects.
- **Internal communication:** Discord.

## Compute and storage

The team does not yet have access to the student GPU server. The mentor asked for an email so it can be forwarded to the technician, who will grant access, possibly as a shared group account. The data should also be transferred to the local NAS for archiving, and the technician should explain how.

## Risks and hurdles

- Metric ambiguity.
- Baseline reproduction depends on IMES's code, expected by Monday.
- **BFloat16 vs. FP16 precision.** BFloat16 keeps FP32's numeric range, but at lower precision than FP16. Results from Colab (FP16) may differ from IMES's app (BFloat16).
- Visual checks so far were done by eye, not yet systematic.
- Human cells are harder to segment than porcine cells.

## Priorities for next week

- Email Nicole: ask why segmentation is needed, request the full code and settings, and align on the integration with the app.
- Preprocessing.
- Find at least three strong candidate models.
- Divide the work among the team members.
- Set up the experiment tracking system.
- Email the mentor to request GPU access and the NAS transfer details.
