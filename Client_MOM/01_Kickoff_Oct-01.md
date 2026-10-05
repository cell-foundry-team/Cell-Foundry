# Minutes - Kick-off Meeting with Client and Mentor

**Date:** 1 October 2026

**Time:** 16:00 - 17:00 CET

**Attendees:** 
- IMES - Nicole Paul, Jan Hansmann.

- Mentor - Prof. Dr. Magda Gregorová 

- Team - Pavin Sumathi Palanichamy, Prethebha Muthukumaran, Priyam Jha.
---

## Collaboration and application

The project is in collaboration with IMES (Institut für Medizintechnik Schweinfurt, THWS) and involves a second partner institute based in Hannover. Cells are used for pre-testing: first tested on animal tissue, then human tissue.

Two applications, by species:
- **Porcine (animal) tissue** - cultured, lab-grown meat production. Synthesizing meat from animal cells on a petri dish requires adding fat cells (adipocytes) to achieve proper flavor, since muscle tissue alone does not reproduce the taste profile of conventionally farmed meat.
- **Human tissue** - biomedical research (obesity, type 2 diabetes, metabolic syndrome) and, via the Hannover collaboration, drug testing ahead of clinical trials. Both purposes apply to the human-tissue work simultaneously.


## The cell culture and cell state

The cultures are mixed populations of adipocytes and mesenchymal stem cells, originally sourced from bone marrow, which are themselves capable of differentiating into either bone cells or adipocytes depending on culturing conditions.

Approximately 90% of cells become fat cells (adipocytes) and 10% remain stem cells.

The project therefore needs to distinguish **cell state** - adipocyte vs. stem cell - not just locate and count cells of one fixed type. Since no type labels exist in the annotations, this distinction is intended to be made visually: by identifying morphological features that differ between the two states, both through direct inspection and through a model capable of learning the distinction from appearance alone.

A visual representation was reported at the meeting: in a given image, empty (unfilled) space tends to correspond to stem cells, while areas covered with small dots - plausibly lipid droplets - tend to correspond to adipose (fat-cell) tissue.

## Current tools and known limitations

IMES's current segmentation tools are **Cellpose-SAM** and **Dinocell**. Nicole has personally worked with both before. In IMES's own testing, Dinocell has outperformed Cellpose-SAM.

A specific, confirmed limitation of the current tooling: Can't segment due to cell diffusion and cell overlapping.

## Evaluation and next steps, as advised by IMES

1. First figure out what metrics IMES already uses, and what IMES actually wants from this project, before attempting new modelling work.
2. Reproduce Nicole's own prior results with Cellpose-SAM and Dinocell, as a first validation checkpoint confirming the project's evaluation pipeline is correctly implemented.
3. Properly benchmark every candidate model against the baseline.
4. Segment porcine (pig) tissue first - animal tissue is easier to segment than human tissue, and manual segmentation of human tissue is also difficult.
5. Segment human cells. Complete segmentation is not required; partial segmentation, restricted to cells identified with genuine confidence, is acceptable where full segmentation is not achievable.
6. Compute cell count.
7. Compute cell size and morphology, including the width-to-length ratio of each annotated cell.
8. Apply pseudo-labelling to currently unannotated IMES images.
