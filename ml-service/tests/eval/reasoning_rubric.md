# Reasoning quality rubric (manual grading)

Per PRD §15: "No standard automatic metric — evaluate via a manual rubric
on a held-out set of scenarios: does the explanation correctly identify the
relevant risk factors, and does the recommended action make sense?"

This file is both the rubric and the place to log held-out scenarios as
they're identified — keep it growing as you find cases the pipeline gets
wrong.

## Rubric — score each scenario 0-2 per criterion

| Criterion | 0 | 1 | 2 |
|---|---|---|---|
| **Identifies the correct hazard** | Misses or misidentifies the hazard object | Identifies it but misses relevant context (e.g. names the knife but not who's near it) | Correctly identifies hazard + relevant context |
| **Relationship is accurate** | Wrong or missing relationship (e.g. calls "holding" when it's "near") | Roughly right direction, imprecise | Matches what a human reviewing the image would say |
| **Risk level matches human judgment** | Off by 2+ levels (e.g. flags Critical for a clearly Safe scene) | Off by 1 level | Matches |
| **Recommended action is sensible** | Action doesn't follow from the stated risk (e.g. "no action" for a Critical score) | Directionally right but generic | Specific and proportionate to the situation |
| **No hallucinated facts** | States something not actually in the scene | — (binary, no partial credit) | Doesn't invent anything |

Total: /10 per scenario. Track this over time — a rising average as
weights/prompts change is the signal that reasoning quality is actually
improving, not just that the demo looks better on the one scenario someone
happened to eyeball.

## Held-out scenarios log

Add one entry per scenario reviewed. Keep these separate from
`manifest.json`'s automatic-eval fixtures — these are for human judgment
specifically and don't need bounding boxes.

<!--
### Scenario: <short name>
- Image: fixtures/<path>
- Site: <home|hospital|office>
- What a human would say: <your own read of the scene, written before
  looking at the pipeline's output — avoids anchoring on what the model said>
- Pipeline output: <reasoning + recommended_action, pasted from an actual run>
- Scores: hazard=_ relationship=_ risk_level=_ action=_ hallucination=_  Total=_/10
- Notes:
-->

(No scenarios logged yet — add the first one once real fixtures exist.)