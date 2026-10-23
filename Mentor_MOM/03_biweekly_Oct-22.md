# Minutes - Biweekly Meeting with Mentor

**Date:** 22 October 2026

**Time:** 15:00 - 16:00 CEST

**Attendees:** 
- Mentor - Prof. Dr. Magda Gregorová

- Team - Pavin Sumathi Palanichamy, Prethebha Muthukumaran, Priyam Jha.

- Guest - Amparo Zamora Mogollo -PhD Student.

---

## Project status and update since 8 October

The team reported on three things: the answers received from IMES (Institut für Medizintechnik Schweinfurt, THWS) to the questions agreed at the last meeting, a review of IMES's Colab notebook, and first zero-shot results of four models with a new fat-area metric. The main change is the goal. IMES's priority is the **fat proportion**, so the project moves from separating individual cells towards measuring the fat area directly.

## Answers received from IMES (email of 15 October)

- **Fat proportion is the first priority.** The area occupied by fat cells matters most for the users. IMES has so far measured it from individually segmented cells and is open to a different method.
- **Cell count is the second priority.** It is important, but not needed right away.
- **Open-source models only.** Any open-source model may be used, on the condition that it can be integrated into IMES's app. The app itself cannot be shared, because it is part of a larger platform that is being extended within an EU project.
- **The Colab notebook is the complete pipeline.** It contains the preprocessing, the metrics with their thresholds, and the loading of the DINOCell base weights. It was shared for reproducibility.
- **The baseline is DINOCell, fine-tuned by IMES,** and not CellposeDINO (called CP-Dino by the team). IMES is not satisfied with its results and finds CP-Dino interesting.
- **No test set.** Only the 28 annotated images were used, split internally into 22 for training and 6 for validation. No separate annotated test images exist. Results on new images were judged visually and found "still not very good". There is therefore no test-set number to reproduce.
- **File names were not used in training.** IMES does not know exactly what "Negativ" means. According to the biologists, all images are bright-field. More annotations are not promised.

## Clarifying the goal: what "fat proportion" means

The team first mixed up two readings: the lipid droplets inside a cell, and the whole fat cell. After discussion with the mentor the team agreed on one definition:
- **Fat proportion is the number of pixels covered by fat cells divided by the total number of pixels of the image.** It is not a proportion of cell counts and not a proportion of diameters. The mentor asked that this definition is written into the project scope.
- The fat cell is taken as a whole, with its lipid droplets together.
- **Cell size and the width-to-length ratio are not needed for now.** Area comes from pixels, not from lengths and widths.
- The annotated masks are the fat cells (adipocytes), marked only where IMES is sure (the team's understanding). Stem cells and other background are neither labeled nor needed.
- IMES may later want the fat share relative to the stem cells and not to the whole image. The mentor expects that IMES has not thought about this yet, and that the team will adapt when it comes up.

## Data and annotations

- **Annotations are partial.** Where it is unclear whether touching cells are one or two, IMES does not annotate. 100% ground-truth accuracy is therefore not possible. A model trained on partial labels may leave fat cells out, and the computed proportion would then be too low.
- **Metrics used by IMES:** Dice, IoU, precision and recall at pixel level, and F1 at an overlap (IoU) of 0.5 or higher at instance level. The mentor clarified that IMES used the validation images for hyperparameter tuning, and that these metrics could serve for testing if test images existed. The team proposes the **relative area error** as the primary metric.
- **File names.** Images named Positiv/Diff have many annotated cells, and Negativ/Ctrl images only a few. The team's reading, taken from the internet and not confirmed by IMES, is that control images are an early stage and Positiv/Diff are matured fat cells. The mentor's hunch is that these are control experiments (nothing induced) against induced ones, and that IMES wants to show the proportion stays low in the control and rises in the induced samples. The method should therefore be **oblivious to the condition**, and the data can be mixed.
- **Patches.** IMES cuts the images into overlapping 512 x 512 windows, so 28 images give many more samples. The mentor noted that this allows a cleverer train/test split, but overlapping parts must never appear in both the training and the test set.

## Review of IMES's pipeline

The team went through the notebook and presented its steps: load the 28 pairs (22 train, 6 validation), improve contrast with CLAHE, turn each mask into arrows that point to the cell centre, start from the pretrained DINOCell with small trainable add-ons (LoRA), train for 1,000 rounds on 512 x 512 pieces, and score every 50 rounds on the validation images to keep the best version.

**Done well:** software versions listed in a config file, most random seeds fixed, one folder per run, split by whole image, and LoRA, which suits very little data. **Not fixed:** PyTorch and Python versions (the Colab default is used, so numbers can change) and GPU determinism.

**Points the team would look into**
1. **Large fat cells get cut short.** Turning arrows into cells follows them for a limited number of steps, with a reach of about 40 pixels. Cells larger than that lose their outer parts, which lowers the computed fat area. In the team's check, about 9% of the cells, holding about a fifth of the fat area, are affected. Proposal: test a larger reach (for example 200 pixels, or a smaller factor such as 1.5 times).
2. **Colour handling.** The original DINOCell converts the image to grey before CLAHE, while IMES applies it to the red, green and blue channels separately. The team's test gave different numbers.
3. **Validation on patches.** Every 50 rounds, 12 patches of the 6 validation images are scored. This is about 25% of each image, mostly around cells, so empty background is rarely checked and false detections are missed. Proposal: score whole images.
4. **No separate test images.** The best version is chosen on the same 6 images that are reported.

The mentor asked the team to judge these steps for themselves: "the fact that they use it doesn't mean it's a reasonable step". The team may use its own preprocessing, as long as the result is an open-source pipeline that IMES can integrate (data in, result out). The arrow mechanism is specific to DINOCell, so other models may need different preprocessing.

## Zero-shot results

Four models were run on whole images with default settings and no preprocessing (25 differentiated porcine bright-field images, true fat area about 19.7% of each image; pp = percentage points, a negative value means too little fat):

| Model | Fat % error | Typical miss | Fat-area Dice | Cell F1@0.5 | Count error |
|---|---|---|---|---|---|
| cpsam_v2 | -6.6 pp | 6.6 pp | 0.66 | 0.43 | +141% |
| cpsam | -4.6 pp | 5.6 pp | 0.65 | 0.47 | +135% |
| cpdino (CP-Dino) | -1.4 pp | 3.7 pp | 0.76 | 0.62 | +58% |
| cpdino-vitb (CP-Dino ViT-B) | -0.1 pp | 4.5 pp | 0.73 | 0.65 | -9% |

- CP-Dino and CP-Dino ViT-B are clearly better than the two CP-SAM models. CP-Dino ViT-B is also much more efficient in GPU memory.
- On six hard images, some models produced up to about 3,300 cells where about 400 are annotated.
- Zero-shot DINOCell performed worse than CP-Dino (and worse than CP-SAM).
- The mentor asked what the "typical miss" and the plus or minus sign mean. They were explained as the average error over the images in percentage points of the whole image, and as predicted minus true.
- On an example overlay (green: IMES annotation, red: model) the model missed cells that IMES had annotated, although the table suggests over-counting on average. The mentor pointed out that this may be a coincidence. The team added that the models also split areas that IMES drew as one large cell into smaller cells.

## Mentor feedback

1. **Define the proportion.** Write into the scope that proportion means the share of image pixels covered by fat cells. The mentor agrees that semantic segmentation should be enough and is easier, because it needs less model capacity and is easier to fine-tune. If no good semantic model is found, go back to the instance models.
2. **Cell count comes second.** Counting separate blobs in a segmentation is a logical problem, not a learning problem. The real difficulty is touching cells that merge into one blob. The mentor advised not to solve it yet and to solve the area first. Panoptic segmentation only adds a class for the background and is not needed now.
3. **Data quality matters most (central advice).** Better data brings more than a better model. The team has about 15 weeks, so it should think about how to get better annotations quickly, which is more about talking to people and exchanging data than programming. The mentor suggested a loop:
   - let the first working model annotate the images and inspect its errors yourselves;
   - where the model finds a fat cell that was not annotated, send these cases to IMES and ask whether the annotation was wrong;
   - correct the data and train again, and apply the same to the unannotated images;
   - IMES then only checks and corrects draft masks, which is much faster than annotating from scratch.
4. **Use disagreement between models.** If several models agree that something is a fat cell and nobody annotated it, it is probably a missing annotation. Where they disagree strongly, the case is suspicious. The mentor compared this to using models as judges.
5. **Choice of models.** The mentor is happy with the proposed models but urged the team to look for more recent ones, since the original U-Net is an old paper (2015), and for models that handle more complex images. A model only sees pixel values, so models pretrained on somewhat different data (for example other cells) may be worth trying. One idea is a two-step fine-tuning: first on a larger public dataset that is closer to the data, then on IMES's data. Models listed in the brief that turn out to be unusable can simply be dropped. Omnipose may be tried, since bacteria images may not look so different. The mentor stressed that all candidates must be open source.
6. **Preprocessing and augmentation.** IMES only mirrors and flips the patches. Many more augmentations are possible and are probably the easiest way to get a better result, because they multiply the training set. When extending the arrow reach, think about side effects, for example that an arrow could reach the centre of a neighbouring cell. Make pictures of the annotations before and after preprocessing to check that the annotations are not destroyed. Make a note of what each step does and judge it as a team, not as IMES.
7. **Experiment tracking.** Weights & Biases in the cloud is fine, and installing it locally is not recommended. It is a skill worth keeping for later projects. The team has to decide which metrics are worth monitoring and should log the hyperparameters and the model setups first, since the default charts can be confusing.
8. **Environments and delivery.** The mentor recommends Docker: one shared container so that everyone works with identical versions, and as the final delivery package for IMES. IMES's own notebook leaves PyTorch and Python unpinned, which is an example of the problem.
9. **Client communication.** Produce first results soon and ask IMES whether the team is going in the right direction. Do not worry about a possible gap between metrics and visual impression before it appears. Requirements may change as the project develops, which is normal.

## Tooling and workflow

- **Weights & Biases:** every run is logged with its configuration, and 50 to 100 experiments are feasible. Only numbers and tables go to the cloud, never images. Each run writes to its own timestamped folder, so nothing is overwritten and every team member can run the same experiment.
- **uv instead of Conda:** faster, works on Mac and Windows, and one command (`uv sync`) installs the agreed versions from GitHub.
- **Scoring script:** one script scores every model with the same metrics (fat percentage, fat-percentage error, foreground Dice, count error, F1). It was checked against a test with hand-calculated answers. The mentor liked the idea and asked for a sketch, and warned that such a test resembles data augmentation and must be set up carefully.
- **Compute:** experiments currently run on the team's own computers.

## Risks and hurdles

- Partial annotations bias the fat proportion, and no model can reach 100% against them.
- No test set exists, so there is no IMES result to reproduce and compare with.
- IMES's pipeline is not fully reproducible (unpinned PyTorch and Python, non-deterministic GPU runs).
- Public microscopy models are often trained on zoomed-in single cells, which look different from the porcine images.
- Touching cells merge into one blob, which makes the count hard.
- IMES may change the definition of the proportion later (relative to stem cells).

## Priorities for the next two weeks

1. Limit the work to at most three core models. CP-Dino (ViT-B is more efficient) is the primary candidate. A semantic model (a recent U-Net variant, nnU-Net or Adipocyte U-Net 2.0) is the main alternative for the area goal.
2. Fine-tune CP-Dino on the data with light adaptation and more augmentation than flips.
3. Refine the preprocessing: test a larger arrow reach, compare channel-wise with grey-first contrast, and compare annotations before and after preprocessing.
4. Do not try to reproduce IMES's result, since there is none. Instead, segment the images, inspect the predictions visually, and also run the model on the unannotated images.
5. Send draft masks to IMES and ask them to check cells that may have been missed in the annotations.
6. Compare several models to find suspected missing annotations.
7. Define a fixed split (18/4/6 or k-fold) that respects overlapping patches, score whole images, and use the relative area error as the primary metric.
8. Agree on one Docker container and select the metrics to monitor in Weights & Biases.
9. Search for more recent semantic segmentation models, and give IMES a short summary together with any further questions.
