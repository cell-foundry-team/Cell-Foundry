# Minutes - Poster Review & Project Synchronization Meeting

**Date:** 11 November 2026

**Time:** 8:00 AM - 9:30 AM

**Attendees:**
- Mentor - Prof. Dr. Magda Gregorová

- Team - Pavin Sumathi Palanichamy, Prethebha Muthukumaran, Priyam Jha.

---

## Purpose of the meeting

This was not the regular biweekly project meeting. The team brought several draft
A1 posters for the 19 November poster workshop and asked the mentor to review them
screen-share style, version by version. A short project-status update (IMES contact,
nnU-Net exploration) was given at the end. The team agreed with the mentor that,
since 19 November is the workshop itself, the usual biweekly meeting that day can be
skipped.

## Poster review: general first impressions

- **Use the space more efficiently.** Looking at the first draft, the mentor's first
  comment was that there was visible empty space that could be used better, and that
  the same cell image appeared three or four times without adding new information.
- **Balance images against text.** The mentor asked the team to weigh how much space
  an image takes against how much information it actually conveys, and to compare
  that against the size of the running text, which in one draft was too small to
  read.
- **Printed size matters.** The poster prints at full A1. In a workshop, people stand
  back from a poster rather than walking up close, so text has to stay legible from
  a few steps away, even for people with below-average eyesight. The team confirmed
  their current text size was barely readable at that distance.
- **Fix, don't just shrink.** The suggested fix was not to shrink images blindly but
  to reduce the size of repeated cell images specifically, and use the freed space
  to grow the body text, for example by moving one image under a "first results"
  heading.

## Poster content review, section by section

### Goal section
- The mentor liked the short, high-impact wording.
- **Repeat phrasing on purpose.** For the porcine and human sentences, the mentor
  advised repeating the same sentence structure word for word ("the aim is...")
  rather than varying it. In research writing, unlike literature, repeated
  structure lets a reader skip the parts that don't change and focus only on what's
  different between the two cases (pig vs. human).
- Showing the pig and human cell images a second time just to say "there's a human
  one and a pig one" was flagged as redundant, since the text already said so; the
  mentor suggested making these images smaller, or using the space for something
  more playful instead (for example the "crunchy meat" idea the team had made as a
  joke version).

### Challenge section
- The mentor's first question was the **image size in pixels** (not just "28
  annotated images"). 28 images only means something once you know how many cells
  are in each: a tiny image with four cells makes 28 very little data; a huge image
  with many cells changes that. The team confirmed the images are about
  1920 × 1080.
- **Simplify every sentence.** Phrases should be as short and plain as possible,
  for example "adjacent cells touch each other, that's fine."
- **Define "dish" once.** The word "dish" was flagged as ambiguous (readers may
  think of food); it needs to be clearly tied to "petri dish" / cell culture dish
  the first time it's used.
- **No articles in titles.** Headings don't need to be grammatically complete
  sentences — drop words like "a" where they aren't needed.
- **Explain "10x" once, properly.** "10x" refers to the bright-field microscope
  setting used to capture the images, not image scale. It should be explained once,
  early, as "10x bright-field," so readers don't misread it as a zoom or size
  factor. Since the project also has phase-contrast images, the poster should make
  clear it is only talking about the bright-field ones it actually works with.

### Approach / pipeline section
- The mentor read through the flowchart out loud (cells → preprocessing →
  models propose → models vote → first result) and liked it: "beautiful," clear
  message.
- The accompanying results graph (comparing models) was judged correct in content
  but too small to read; the team should make it bigger or increase contrast
  between the colors used (for example gray vs. orange) so people can tell them
  apart without reading labels closely, even if that means the image stays large.
- The small illustration meant to explain "voting" among models was judged
  acceptable, but also flagged as too small given how much it's trying to convey.

### References / next steps section
- The mentor pointed out this section largely repeated "models vote, experts check,"
  which is already shown in the pipeline, so it added little new information.
- Suggested replacement: list **which models were actually tried** (as a kind of
  reference list), or something about the technical solutions used, rather than
  repeating the pipeline.

## Comparing the different poster drafts

Three to four different visual drafts were shown and compared directly:

1. **The "professional" structured design** (phases/boxes: data review, image
   quality, shared protocol, zero-shot test, fine-tuning, etc.). The mentor liked
   that it had a clear phase structure, but found the wording hard to parse on
   first read (jargon like "shared protocol," "zero-shot test" without enough lead-
   in), and noted it was **missing the feedback/data loop** entirely.
2. **The simpler design** (short, plain labels: "models propose," "models vote,"
   "data"). The mentor liked that this one read as more obviously written by the
   team rather than AI-generated, and that it already showed the feedback loop
   (the ensemble / active-learning cycle), which the first design lacked.
3. **A "fun" version** (the hamburger / crunchy-meat idea) and a **fully AI-
   generated draft** were also shown. The AI-generated one was judged easy to read
   because of its bullet points, but the mentor was explicit that genuine,
   team-written text is preferred over AI-generated text, even if the content
   overlaps, because it reads as more honest and the team can actually defend every
   sentence.
4. A **render-style design** with a lot of empty space was also shown and rejected
   for the same "use the space better" reason as the very first draft.

**Decision:** keep **one single visual style** — do not mix two different poster
styles together, since a team has to agree on one direction rather than Frankenstein
two designs. The mentor's own preference leaned toward the structured/professional
design for overall look, but recommended folding in the better-written, more genuine
content and the feedback-loop diagram from the simpler draft. Concretely: keep the
image and preprocessing steps as understood/obvious and not over-explained, but make
the feedback loop very visible, for example as a "zoomed-in" detail box opening out
of the main pipeline to explain the active-learning / ensemble-labels loop
specifically.

### The "Lessons Learned" box
- The mentor explicitly said this is probably **the most important box on the whole
  poster** and should not be removed, even while trimming other sections.
- As drawn, the box was judged too large and the message inside too dense/unclear.
  The mentor pushed the team to restate the finding as one simple, plain-English
  sentence rather than four cell-overlay images: *"Zero-shot overestimates fat
  cells at the start (control images) and underestimates them later
  (differentiated images). Fine-tuning fixes both."* This single sentence replaces
  the need for the extra image row entirely, and can optionally be illustrated with
  the same gray/orange coloring already used in the bar chart (e.g. gray =
  "overestimates," orange = "gets it right").
- The "four images" version the team had prepared (zero-shot vs. fine-tuned, on
  both a control-day and a differentiated-day image) was discussed at length: the
  mentor understood the intended message but repeatedly pushed the team to state it
  as one clear sentence instead of leaving the reader to infer it from four small
  panels.

### Control images and the real research goal
- A separate, more fundamental point came out of this discussion: the poster
  nowhere explains that IMES's actual goal is to measure **how the fat percentage
  changes over time** (e.g. from day 0, when the drug is introduced into the
  dish, to roughly day 20+, once cells have differentiated), not just to measure
  fat percentage on a single image. "Control" images are the day-0 baseline
  (almost no fat cells), and the differentiated images are the later time point.
- The mentor said this had never been explained anywhere in any draft and that
  they had only understood it by inference while reading the posters. This should
  be stated explicitly, ideally very simply, for example "Day 0: 0% fat cells →
  Day ~20: ?%."
- Agreed fix: keep a control-image example and a differentiated-image example,
  explicitly labelled as before/after a roughly 20-day culture period (exact day
  count doesn't need to be precise — "20+ days" or "~20 days" is fine, since nobody
  will fault the team for not knowing the exact number).
- The mentor also recommended writing titles more like real poster titles — using
  symbols (question marks, exclamation marks, arrows) and short phrases rather than
  full sentences, for example turning "Which culture conditions produce more meat?"
  into something even shorter with symbols.

### Human cells
- Since the project is not primarily working on human cells yet, the mentor
  suggested moving the human-cell content out of the main body and into a "Future
  Work" section instead, shown as a teaser image with an honest note that it will
  only be attempted after the cultured-meat (porcine) work is done. This also frees
  up space used by near-duplicate pig/human image pairs.

### The closing "hamburger" image
- The mentor's one firm visual suggestion: end the poster with the "hamburger"
  image from the fun draft, replacing a more literal diagram. None of the drafts
  currently state outright that the end goal of the porcine work is cultured meat;
  a hamburger image makes that connection immediately and gives people a reason to
  stop and look ("how do you get from a microscope to a hamburger?"). This is
  purely for visual pull — the mentor does not want to adopt the fun draft's own
  structure or wording, only this one closing image, placed into the otherwise-kept
  structured design.

### Using the freed space
- Posters are expected to be dense — the mentor's advice was not to delete content
  to make room, but to remove **surrounding white space** around existing boxes so
  the added content (more of the Lessons Learned detail, the loop diagram, model
  references) has somewhere to go, without making any individual block more
  cramped than the others. A poster where one block is visibly denser than its
  neighbors is harder to read.

### "Current status" framing
- The mentor liked the team's "Done" / "In process" labelling idea and encouraged
  it, reassuring the team that a static printed poster showing a work-in-progress
  state is completely normal — anyone reading a hallway poster already understands
  it reflects one moment in time, with work continuing before and after.

## Project status (outside the poster)

- Prethebha is sending IMES a document explaining how to access the review
  material and give feedback, following on from Nicole's agreement to review
  candidate annotations.
- In the meantime the team is exploring **nnU-Net** as the semantic-segmentation
  model.
- The team confirmed that, given the amount of poster work still needed, the data-
  versioning cleanup discussed below will happen after poster submission, not
  before — the mentor agreed this is fine and should simply be reflected honestly
  in the poster's "current status" / "future work" framing (e.g. "now waiting for
  the first annotation data to come back").

## Mentor feedback: data versioning (important, for after the poster)

1. **Version the annotation data carefully.** Once IMES starts sending back
   corrected/approved annotations, the team will have at least three distinct
   versions in play: the original annotations, the version sent to IMES, and the
   version IMES returns. The mentor warned this is a classic place for a project to
   turn into "a total mess" if not tracked properly, especially later when trying
   to show how annotation quality affected model results.
2. **Use dates, not version numbers.** Rather than "version 1 / version 2 / version
   3," prepend filenames with an ISO-style date (e.g. `2026_11_10_...`), year
   first, so files sort chronologically by default. This was the mentor's advice
   "after years of experience": unless you have a proper system like GitHub, a
   prepended date is the simplest reliable way to order actions.
3. **Keep a protocol/log.** Alongside the dated files, keep a short README or
   spreadsheet recording what action happened on which date ("this is when we first
   sent data to IMES," "this is when we got it back"), not just the fingerprint
   itself.
4. The team already has a version of this in progress: a target-image list for
   IMES, plus a spreadsheet tracking which image each change corresponds to and how
   file names map between what was sent and what comes back. The mentor's main
   addition was the date-prefixing convention for the files themselves.

## Priorities before 19 November

1. Keep one single poster style; do not mix two designs together.
2. Rework the Goal section: repeat sentence structure between porcine and human
   cases; shrink or drop the duplicate cell images there.
3. Add the pixel size of the 28 annotated images to the Challenge section; simplify
   wording; define "dish" clearly; explain "10x bright-field" once.
4. Keep the pipeline diagram, but make the feedback / active-learning loop clearly
   visible, e.g. as a zoomed-in sub-diagram.
5. Replace the four-image "zero-shot vs. fine-tuned on control vs. differentiated"
   panel with one plain sentence (overestimates early, underestimates later; fixed
   by fine-tuning), optionally color-coded like the bar chart.
6. Add an explicit control-vs-differentiated / day-0-vs-day-N explanation of what
   the fat-percentage change actually measures.
7. Move human cells into a "Future Work" teaser, honestly framed as attempted only
   after the porcine/meat work.
8. Replace the final pipeline/structure image with the hamburger image, kept inside
   the structured design.
9. Remove white space around boxes rather than shrinking the "Lessons Learned" box;
   that box must stay.
10. List the specific models tried (references) instead of repeating the pipeline
    in the final section.
11. Keep the "Done" / "In process" status framing on the poster.
12. No biweekly meeting is scheduled for 19 November, since that is the poster
    workshop itself.
13. After the poster is submitted: start the date-prefixed data-versioning scheme
    for annotation files and keep a protocol/log.
