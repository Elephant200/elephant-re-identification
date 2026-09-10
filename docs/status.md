# Status

The standard AlphaPhant composition is implemented and verified.

## Next steps

### 1. Curate images

Adapt the [image picker](../apps/image_picker/README.md) with heuristics/AI ranking and human approval. Automatically reject only unusable files and clear duplicates; keep candidates recoverable and spot-check rejected/low-ranked images. Inventory available data, then fix quality criteria, cohort targets, and an annotation budget before bulk curation or new benchmark results.

Tuning and benchmark sightings need sighting ear pairs. Model data can contain additional, unpaired, side-unbalanced photos.

Enlarge the benchmark to **100–200 elephants with 3–5 approved sightings per elephant**, first securing 100 elephants with three approved sightings each. Protect sufficient clean repeated-sighting identities for that target first, preferentially using identities without human annotations; use annotated identities if image scarcity requires it. Then choose a smaller representative tuning cohort and direct annotation-rich and retrieval-ineligible identities toward model data. Seek good images in every split.

Calibrate AI photo-quality review against 100–200 user-graded ear examples; a Roboflow VLM is a candidate. Freeze and retain per-photo grades for reuse across expansion and annotation selection, and derive elephant-level quality summaries from them. Interior occlusion is acceptable when the relevant outer boundary and both anatomical endpoints remain clearly visible.

Run grading as a one-time batch using the existing cache, without a grade-editing interface or override infrastructure. Each photo has independent ear-side scores and its own best-ear image score; one poor or absent opposite ear does not reduce a clean ear's score. Derive sighting quality from the weaker of its best left/right choices and elephant retrieval quality from its third-best sighting pair. Retain good-image and annotation counts for model-data selection.

Human-annotated photos of benchmark elephants cannot train models, even across different sightings. Include suitable annotated photos in the benchmark to study whether better segmentation is associated with better retrieval accuracy. Complete each sighting ear pair with a suitable opposite-ear photo from the same sighting. Update existing sighting entries rather than duplicate them.

Where a suitable annotated photo belongs to an existing benchmark sighting, switch that side's selection to the annotated photo after verifying its source mapping. AI assists photo-quality review, especially occlusion assessment, before human selection. The selection page shows raw ear images without numerical quality scores. Save incomplete selections and support undo; export only complete validated retrieval pairs. See the evolving curation specification in `.scratch/image-curation/spec.md`.

### 2. Freeze splits

**Sightings are indivisible:** a different photo from the same sighting still invalidates that sighting's use in a separate split. Also keep elephants disjoint across model, tuning, and benchmark cohorts. The benchmark requires 3–5 approved sightings per elephant; counts alone do not establish photo quality or eligibility. Model and tuning allocation depends on the remaining data.

Preserve benchmark membership except Lotter, whose removal from the expanded benchmark is approved because only two source sightings exist; retain the historical manifest for reproducibility. The entirely AI-selected tuning set is flexible and may be rebuilt before freezing. Freeze private manifests for each cohort and each model's train/validation/test splits, with identity-disjoint roles coordinated across models. Crops, labels, and augmentations inherit source assignments. Audit duplicates and prior training exposure; record permanent IDs, counts, versions, and hashes.

### 3. Annotate

- **Localization:** SAM3-generated labels with quality review. SAM3 is imperfect; human-check held-out labels rather than measure agreement with SAM3 alone.
- **Segmentation:** the user manages human masks through an external service. Prefer new annotation expenditure on model training images; reuse existing benchmark masks for the planned segmentation-quality study. Benchmark-elephant annotations never train models.
- **Landmarks:** older YOLO26 drafts or, potentially, drafts from another foundation model, reviewed/corrected by a human; use human-reviewed validation/test labels. Distinguish any machine-only training labels.

Validate crop/full-image coordinates and retain annotation provenance. The export at `/Users/alex/Downloads/30997` contains **500 annotated image records across 461 sightings**. Whole-elephant metadata reconciliation places **132 records in benchmark identities, 295 in tuning identities, and 73 in neither** under current assignments. Tuning may be reallocated. Exact sighting overlap is narrower: 40 benchmark and 90 tuning sightings, totaling 146 records. Forty-two annotated filenames also occur in prior landmark dataset split directories; actual weight-training exposure remains to be checked. Crop mappings and duplicate checks remain outstanding.

### 4. Train and tune

Train **YOLO26 localization**, compare **U-Net/BiRefNet segmentation**, and train **heatmap landmarks**. Select on model validation data and evaluate locked model test splits. Integrate through existing inference interfaces with versioned weights and producers. Tune AlphaPhant only on tuning data, then lock the complete configuration.

### 5. Benchmark

On the expanded existing benchmark, compare AlphaPhant with current versus new preparation, then AlphaPhant/CurvRank/MiewID with shared preparation and identical query/candidate sets. Predefine a small set of ablations and the human-mask analysis of segmentation quality versus retrieval accuracy. Account for repeated sightings within elephants; association alone does not establish causation. Automated retrieval uses predicted masks and landmarks.

Report top-1/top-5, paired uncertainty, actual catalog sizes, ties, and preparation failures using [evaluation.md](evaluation.md). Preserve full scores and reproduction metadata; never silently drop failures or use final results to select models or parameters.

### 6. Write the paper

Report curation, splits, annotation provenance, training, comparisons, segmentation-quality associations, and limitations of the curated high-quality-image setting. Prepare figures, tables, and reproduction instructions from the frozen experiments.
