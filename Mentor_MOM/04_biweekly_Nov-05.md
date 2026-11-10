# Minutes - Biweekly Meeting with Mentor

**Date:** 5 November 2026

**Time:** 16:00 - 17:00 CET

**Attendees:** 
- Mentor - Prof. Dr. Magda Gregorová

- Team - Pavin Sumathi Palanichamy, Prethebha Muthukumaran, Priyam Jha.

- Guest - Amparo Zamora Mogollo -PhD Student.

---

## Project status and update since 22 October

The team reported on five things: a first active-learning cycle with a review tool for the IMES expert (Institut für Medizintechnik Schweinfurt, THWS), three points on how to report the fat-area error, a check of the licenses of the candidate models, observations on IMES's data and code, and first fine-tuning results with an ablation of IMES's contrast step (CLAHE). The main outcomes are that fine-tuning clearly improves all models, that CLAHE does not help and is being dropped, and that the mentor sees the next large gains coming from better data and from using several models together.

## Active learning: first cycle and review tool

- **Starting point.** From the last meeting the team took the advice to make the data better. If IMES annotated about 300 fat cells in an image and the model finds about 120 of them, the remaining 180 may be fat cells, background or stem cells. The proposal is to let an expert check them.
- **Method.** Two models (CP-Dino ViT-B and CP-Dino ViT-L) predict cells on their own. Predictions that match an IMES annotation count as confirmed fat cells. Predicted regions that nobody annotated are the base candidates. Candidates on which both models agree are the final candidates for the expert.
- **Review tool.** Because the data are confidential, the team built a password-protected HTML file in which the expert can approve or reject a candidate, adjust the yellow outline (for example when two cells are predicted as one) and draw. The decisions are sorted into approvals and rejections, with image names kept. Approved cells are meant to be added to the training data for a retrain.
- **Status.** The tool was finished on the day of the meeting and has not yet been sent to IMES. It is not known whether IMES is willing to do the review. How the file reaches the expert and comes back (hosted on the team's Cairo server, or sent as a file and returned zipped) is still to be agreed with IMES.
- **Reaction.** The mentor likes the strategy of two competing models that agree on what is annotated and on what is a candidate to verify, but criticised the process (see mentor feedback 1).

## Reporting the error: three points raised by the team

1. **IMES annotations as a lower bound.** The annotated fat percentage covers only cells IMES is sure about, so the team proposed treating it as the minimum a model should reach, and inspecting images where a model predicts more.
2. **Signed errors cancel out.** If one image is over-predicted by 4 points and another under-predicted by 4, the mean error is 0 although the model is wrong on both. In the example shown for zero-shot CP-Dino, the signed average was about 0.5 pp, while the absolute error was 5.8 pp. The team proposes to report the absolute error. The mentor agreed.
3. **Licenses** (see below).

**The mentor's response to point 1:** the statement is not as black and white as presented. IMES say themselves that the annotations are not perfect, so the model may be right where it predicts less. A hard rule that the model may never go below the annotation could make the model worse. Also, annotators work with a brush and easily go over the cell edge, which can inflate the annotated area. If the team wants to use the lower bound, it should be treated as a stated assumption and tested by training one model with and one without it.

## Data observations

- **Same ID for two regions.** In a few ground-truth masks, two regions at least 100 pixels apart carry the same cell ID. This does not affect the fat percentage but does affect cell counts.
- **One large cell versus several.** IMES sometimes annotated a very large cell as one, where the model predicts three. The counts then differ although the fat percentages can be similar. The mentor's advice: report this as it is, and in the final report give IMES examples, for instance that the model's counts are lower or higher because of such cases.
- **IMES's cropping.** IMES cuts the images into windows that are always centred on a random annotated cell (centre between 128 and 384 pixels). The background is therefore rarely seen, which hurts control images (start of the culture, almost only background). The team is not using this step and trains on whole images.
- **Hidden behaviours in IMES's code** found while cleaning it: an unknown model name silently falls back to CP-SAM v2, training images with five or fewer cells are silently dropped (the control images), and all random seeds produce the same crops. The team removed the fallback, keeps the control images, and now lets the crops differ per seed.

## Licenses

- **CP-SAM** is non-commercial, yet performs well. **CP-Dino and the Dino ViT models** have their own license: ship the license text, credit the source and avoid military use.
- **nnU-Net and Micro-SAM** are the cleanest to deliver, because they can also be used commercially.
- **Mentor's explanation:** put the license text and the source at the end of the README of the GitHub repository. A project built on a non-commercial tool cannot be licensed more openly than the tool, only more restrictively. The license also states what the software may be used for and removes the liability from the authors. As academics the team can use the models, but must state the license clearly in the slides and the README, and make IMES aware, since the decision on commercial use (for example lab-grown meat) is theirs.
- **Check first.** The mentor advised to check with IMES before relying on a restricted model.

## Ablation studies and the IMES pipeline

- **The team's ablation.** Following IMES's pipeline, the team tested each model on raw images and with CLAHE, zero-shot and fine-tuned. The mentor confirmed this is an ablation study: one decision (here the contrast step) is tested by training two models and comparing them. If the new idea is not better, the initial method stays and the result is still worth reporting.
- **Experiment layout.** The 12 scorings for CP-Dino ViT-B are: zero-shot raw and CLAHE, fine-tuned raw and CLAHE, with three seeds each. Every team member currently logs to their own Weights & Biases space.
- **Shared split.** IMES had no train/validation split. The team agreed on 18 training, 4 validation and 6 test images, built from the 28 annotated images and spread over the conditions (control, differentiated, and hard images). A fingerprint of the split is checked automatically so that every model uses exactly the same images and no test data is ever seen.
- **Three seeds.** Each model is trained three times with different seeds. The reported plus or minus values are over these three runs.
- **Same strategy as IMES for now.** To get a fair first comparison, the team kept IMES's crop strategy, CLAHE settings and LoRA fine-tuning (about 400 rounds instead of 1,000 to save time, with a validation set to choose the stop). Other crop strategies are to be tested later.

## Fine-tuning results so far

- **CLAHE does not help.** With and without CLAHE the results are about the same, and the model is slightly better without it. The team proposes to drop it, which makes the code simpler.
- **Fine-tuning helps a lot.** For CP-Dino ViT-B, the zero-shot fat-area miss on raw images was 7.9 percentage points and fine-tuning reduced it clearly.
- **Control images.** Zero-shot models predicted roughly 300 to 400 cells on control images with only a few fat cells. After fine-tuning this fell to 4 to 10 cells with CLAHE and 5 to 7 without, over the three seeds.
- **Models compared.** CP-Dino ViT-B (more efficient, fits in about 4 GB, run on a laptop with 8 GB VRAM), CP-Dino ViT-L (run on the Cairo server), CP-SAM (run on the server by Pavin), and IMES's DinoCell. After fine-tuning, the three chosen models perform clearly better than DinoCell. CP-SAM performs similarly to the others after fine-tuning, although IMES had reported it as the worst.
- **Results not complete.** The CP-SAM and DinoCell result files could not be copied from the server to the NAS, so some CP-SAM numbers were copied by hand and the zero-shot columns are from 18 October. A third model's results will be uploaded to GitHub once the server problem is solved.
- **Not yet available:** nnU-Net (the only semantic model with a commercial license), which takes long to train. Its results are planned for the next meeting. IMES has not given any results of its own to compare with, only an unexplained "70%".

## Mentor feedback

1. **Check before building a tool.** Do not invest time in a tool before knowing whether the other side is willing to use it. The review tool may be a waste if IMES has no capacity, and the password was not needed. Before building anything, get a basic understanding of what IMES is willing to do, by call, email or conversation, and talk to them like normal people. The mentor liked the strategy itself.
2. **Contacting IMES.** Ask Nicole whether the expert review is possible. If she declines, tell the mentor, who will then talk to Jan Hansmann and suggest the tool for other projects as a fallback.
3. **Report the error well.** Absolute error is a reasonable fix. Better still, plot a histogram of the errors over all images, plus summary statistics such as the median or standard deviation, so that the full spread is visible. Then examine the worst cases visually: failure cases show what is wrong with the data or annotations and are where the largest gains come from.
4. **Licenses.** Ship them in the README, make IMES aware, and check with IMES before using restricted models.
5. **Ablations by filtering.** Do not run every ablation against every model and seed (12 runs times two crop strategies is already 24). Decide step by step. Test CLAHE on one model, confirm on a second if needed, then drop it and test the next idea (for example the crop strategy) only on raw images. If the result is obvious, there is no need to repeat it on the third model.
6. **Crop strategy.** IMES's argument is that a segmentation model needs something to segment, and empty images may confuse it. The team's argument is that the background is needed. Neither is obviously right, so try both as an ablation.
7. **Cross-validation.** Not needed now. Use 5-fold cross-validation at the very end, to prove the final choice among a few plausible competitors. The three seeds already give some of its benefit, provided each seed picks different crops. The team should count how many crops each of the training, validation and test sets really contains.
8. **Use the models together.** The mentor encouraged the team to keep several models, let them vote, and use the agreement as pseudo ground truth on the unannotated images (about 100 of them), without waiting for IMES. If a majority (for example 50% or more) agrees, treat the cell as positive, add it to the training and validation data, retrain, and keep the original annotated test set untouched to measure whether the automatic annotation improves the result. Contours will not overlap exactly, so a rule for merging them (some kind of average) is needed. If it works, IMES can reuse the script on any future unannotated images, and the final delivery can be the ensemble of models rather than one model. This is common practice (ensemble modelling, as in winning Kaggle solutions). Human cells and the porcine phase-contrast images can be tried the same way.
9. **Data management.** Once data are added or changed, record exactly what the data were at each stage. Never overwrite the original data. Keep the fingerprints of each data version and the model weights, so that every step can be traced.
10. **Improve the pipeline.** The baseline (IMES's pipeline, re-implemented with other models) is now frozen, so the team is free to change it, in particular to add augmentation. A method tested on one model (for example ViT-B) can be shipped as a script, and it will very likely transfer to ViT-L and CP-SAM.
11. **Poster.** The poster is a visual prop to support the explanation, not a place for all information. It should show the overall goal, the specific challenge, and the general strategy (several models, active learning, automated active learning) at overview level, not in depth. The mentor asked why three images from CP-Dino and CP-SAM would be needed, and recommended using the images the team already has.
12. **Communication.** Slow down, start from the problem and from what IMES wants (the fat percentage), define terms and use examples with different numbers. A listener who does not know the project should be able to follow.
13. **Server and NAS.** Write to Leon as soon as possible to solve the transfer of results from the server to the NAS.

## Tooling and workflow

- **Weights & Biases:** useful for ablations because a run with and without a component can be set up directly. A shared project is planned for the final runs only, to avoid a mess.
- **Scripts:** shared with all team members, with the same preprocessing, split and scoring for every model.
- **Compute:** CP-Dino ViT-B on a team laptop (8 GB VRAM), ViT-L and CP-SAM on the Cairo server.
- **Work split:** one member improves the pipeline (augmentation, preprocessing) using ViT-B because it is fast. Others work on ensembling and on active learning and build on that script.

## Risks and hurdles

- IMES may not be willing or able to review the candidates, so the review tool might not be used.
- Results from the server could not be recovered, so some comparisons rest on numbers seen but not saved.
- IMES's annotations are partial and imperfect, so error numbers against them have limits and some model predictions may be correct where the annotation is not.
- Cell counts differ systematically because the model splits large cells that IMES drew as one.
- Combining the masks of several models into one training mask needs a careful merging rule.
- Changing the data during the project needs strict versioning to keep the comparison with the frozen baseline fair.
- License limits of CP-SAM and CP-Dino for commercial use.

## Priorities for the next two weeks

1. Ask Nicole whether the expert review is possible, and tell the mentor if she declines.
2. Write to Leon about the server and NAS transfer and recover the missing results.
3. Report the errors as absolute error with histograms, and examine the worst cases visually.
4. Drop CLAHE, confirm on a second model if needed, and run the crop-strategy ablation on raw images only.
5. Count the crops in the training, validation and test sets.
6. Add more augmentation to the pipeline, testing on ViT-B first.
7. Build the ensemble and automatic annotation of the unannotated images (voting rule and merging rule), keeping the test set untouched and the data versions recorded.
8. Put the license text and sources in the README and tell IMES about the license limits.
9. Train nnU-Net and present its results at the next meeting.
10. Prepare the poster as an overview of goal, challenge and strategy.
