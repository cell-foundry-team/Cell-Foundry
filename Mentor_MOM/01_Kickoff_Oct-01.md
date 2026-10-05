# Minutes — Kick-off Meeting with IMES

**Date:** 1 October 2026
**Attendees:** IMES — Nicole Paul, Jan Hansmann. Team — Pavin Sumathi Palanichamy, Prethebha Muthukumaran, Priyam Jha.

Note on communication: Nicole does not speak much English; Jan's English is workable. Written follow-up is likely more reliable than live spoken exchange with Nicole specifically.

Compiled from two independent sets of notes taken during the meeting (Pavin/Prethebha, and Priyam), merged into one account.

---

## Collaboration and application

The project is in collaboration with IMES (Institut für Medizintechnik Schweinfurt, THWS) and involves a second partner institute based in Hannover. The Hannover collaboration concerns cell culture for drug testing ahead of clinical trials. Cells are used for pre-testing: first tested on animal tissue, then human tissue.

Two applications, by species:
- **Porcine (animal) tissue** — cultured, lab-grown meat production. Synthesizing meat from animal cells on a petri dish requires adding fat cells (adipocytes) to achieve proper flavor, since muscle tissue alone does not reproduce the taste profile of conventionally farmed meat.
- **Human tissue** — biomedical research (obesity, type 2 diabetes, metabolic syndrome) and, via the Hannover collaboration, drug testing ahead of clinical trials. Both purposes apply to the human-tissue work simultaneously.

One detail reported but not yet resolved: the cultured-meat product was described as "soy meat." What this term precisely denotes is unclear and worth asking IMES directly.

## The cell culture and cell state

The cultures are mixed populations of adipocytes and mesenchymal stem cells, originally sourced from bone marrow, which are themselves capable of differentiating into either bone cells or adipocytes depending on culturing conditions.

Approximately 90% of cells become fat cells (adipocytes) and 10% remain stem cells. This ratio was reported at the meeting and independently recorded by two note-takers, but it is an estimate, not a fixed or required target, and is not computable from the current annotation data (which carries no cell-type labels).

The project therefore needs to distinguish cell **state** — adipocyte vs. stem cell — not just locate and count cells of one fixed type. Since no type labels exist in the annotations, this distinction is intended to be made visually: by identifying morphological features that differ between the two states, both through direct inspection and through a model capable of learning the distinction from appearance alone.

A visual heuristic was reported at the meeting: in a given image, empty (unfilled) space tends to correspond to stem cells, while areas covered with small dots — plausibly lipid droplets — tend to correspond to adipose (fat-cell) tissue. This was checked directly against a small sample of the project's own data on 5 October and holds up well on visual inspection, though not yet confirmed systematically or quantitatively across the full dataset.

## Current tools and known limitations

IMES's current segmentation tools are **Cellpose-SAM** and **Dinocell**. Nicole has personally worked with both before. In IMES's own testing, Dinocell has outperformed Cellpose-SAM.

A specific, confirmed limitation of the current tooling: it is not able to correctly segment overlapping or intersecting cells.

## Evaluation and next steps, as advised by IMES

1. First figure out what metrics IMES already uses, and what IMES actually wants from this project, before attempting new modelling work.
2. Reproduce Nicole's own prior results with Cellpose-SAM and Dinocell, as a first validation checkpoint confirming the project's evaluation pipeline is correctly implemented.
3. Properly benchmark every candidate model against the baseline.
4. Segment porcine (pig) tissue first — animal tissue is easier to segment than human tissue, and manual segmentation of human tissue is also difficult.
5. Segment human cells. Complete segmentation is not required; partial segmentation, restricted to cells identified with genuine confidence, is acceptable where full segmentation is not achievable.
6. Compute cell count.
7. Compute cell size and morphology, including the width-to-length ratio of each annotated cell (requested by Prethebha's notes as a specific statistic worth tracking).
8. Apply pseudo-labelling to currently unannotated IMES images.

A number was mentioned — "at least 70%" — attached to an unspecified metric. Neither note-taker could confirm what this percentage measures, and the team has dropped it as a tracked item pending clarification; it should not be treated as an actionable target until IMES clarifies what it refers to.

## Open questions for IMES / Hannover

- What exactly "soy meat" denotes.
- What results Nicole's prior work with Cellpose-SAM and Dinocell actually produced, so "reproduce" has a concrete target.
- What metric the "at least 70%" figure refers to, if it is still relevant.
- Whether the empty-space/dot visual heuristic for stem-cell vs. adipocyte holds up systematically across the full dataset, both species, and both imaging modalities.

## See also

Full detail, including the exhaustive reconciliation against the project's real annotated data, is recorded in `Complete Notes.md` §9 (scope pivot) and §9.12 (Priyam's independent minutes and the same-day visual-heuristic check).
