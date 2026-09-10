# Future research

Deferred research directions. These are not current implementation requirements.

## Inference refinements

Explore improved pose, visibility, or contour-quality estimation beyond the planned model training and image-selection work.

Replace shared `Detection` returns with typed ear-segmentation and ear-landmark results when revisiting the semantic inference interfaces in [architecture.md](architecture.md).

## Additional identity signals

Tear profiles are interpretable but cannot represent every useful identity feature. Later research may investigate:

- learned ear or part embeddings;
- depigmentation, vein, scar, or texture descriptors;
- local feature matching;
- holes when suitable annotations and segmentation exist;
- tusk or body evidence when an experiment justifies reintroducing it.

Additional signals should earn inclusion through separate evaluation.

## Broader retrieval

Later work may explore:

- one-sided queries;
- open-set rejection for elephants absent from the catalog;
- approximate retrieval for much larger catalogs;
- temporal generalization across fixed observation periods;
- uncertainty estimates across repeated sightings.

These extensions should preserve the distinction between similarity-based candidate ranking and a final identity decision.

## Field application

Extend research image selection into a field evidence-review and identity-decision workflow; see [reference/application.md](reference/application.md). The application should use the shared domain objects, PhotoStore, and AlphaPhant pipeline after ear selection.

Application import may later add duplicate detection or content identity across external systems. Permanent opaque IDs and immutable original-photo semantics remain shared with the research system.
