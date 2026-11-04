# Active learning, cycle 0: porcine bright-field

What this is: cells that two zero-shot models (cpdino-vitb and cpdino) find in places where the existing
annotations have no cell. They are PROPOSALS, not new ground truth. Nothing here is confirmed by an expert.

Files
- review.csv            one row per candidate (100); fill the `decision` column with yes / no / partial
- thumbnails/Cxxx.png   256 px crop shown at 2x. GREEN = existing annotation. YELLOW = candidate both models found.
- candidate_masks/      full-size label images (<image>_candidates.tif); the pixel value is the candidate number (C001 = 1)
- reports/              summary.json, per_image.csv, contact_sheet_top20.png
- al_cycle0.py          the script that produced this (needs cellpose 4.2.1.1, weights for cpdino and cpdino-vitb)

How to read review.csv
- models: `both` = both models found it (most likely a missed annotation)
- conf_mean: mean cell probability of the model(s)
- bbox_x/y/w/h: location in the full 1920x1080 image; crop_x/y: top-left of the thumbnail
- split: only `train` and `val` images are included. The 6 test images of split porcine_bf_v1 are excluded on purpose.

Decision meaning: yes = a real fat cell that should be annotated; no = not a fat cell (stem cell, debris);
partial = a fat cell but the outline needs correction.

To add a confirmed cell to a mask: take the pixels where candidate_masks/<image>_candidates.tif equals its number
and write them into the full-size instance mask at the same pixel coordinates.

Known limits
- Candidate = less than 10% of the predicted cell overlaps an annotation and the area is at least 150 px.
- Zero-shot models over-detect; many candidates will be stem cells. The control image contributes many.
- No retraining has been done. Impact Precision cannot be computed until someone reviews the candidates.

## Sending the candidates to the expert (password-protected review page with a drawing tool)
- thumbnails_layers/      per candidate: Cxxx_base.png (clean picture), Cxxx_gt.png (existing annotations, green), Cxxx_cand.png (proposed outline, yellow)
- build_review_page.py    builds ONE encrypted HTML file (AES-256-GCM, key from your password). No server, no upload, no network calls.
- export_accepted_masks.py  turns the expert's decisions.csv + edits.json into full-size masks of accepted cells
- sort_decisions.py       puts the thumbnails into approve/ and reject/ folders and prints the approve rate (uses thumbnails_for_expert/)
- demo/                   a version with FAKE images to test the page (password in the chat, not real data)

    ~/cf_work/cf_env/bin/python build_review_page.py --cycle-dir . --out ~/review_cycle0.html     # asks for a password (12+ characters)
    ~/cf_work/cf_env/bin/python export_accepted_masks.py --edits edits.json --cycle-dir . --out accepted_cycle0
    ~/cf_work/cf_env/bin/python sort_decisions.py --decisions decisions.csv --cycle-dir . --out sorted_cycle0

The expert can RESHAPE the yellow outline (Nudge tool: drag the edge, like pushing clay; slider = how much of the line moves) and DRAW new outlines.
Rules (see export_accepted_masks.py): approve + reshaped outline = it replaces the yellow one; approve + drawn outline (no reshape) = it replaces the yellow one;
reject + drawn outline = extra fat cell nobody marked. edits.json (version 2) holds `adjusted_outline` (reshaped) and `outlines` (drawn), in full-image pixel coordinates.
Limits: points cannot leave the 256 px window of the picture, so a cell cannot be extended past the image border or past what the picture shows.
The editable outline is a 64-point approximation of the model mask (layers/Cxxx_poly.json).
Every build has a new random salt, so a rebuilt page starts with a clean slate. The page also has a "Clear all my answers" button.
Send the HTML through THWS Cloud and the password through a DIFFERENT channel. Never commit the HTML (it contains the images).
