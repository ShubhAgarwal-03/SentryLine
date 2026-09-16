{
  "_comment": "Harness smoke test only — NOT real accuracy data. See README.md. Run generate_synthetic_fixtures.py first to create the referenced images.",
  "entries": [
    {
      "image": "_synthetic_smoketest/smoketest_1.jpg",
      "environment": "home",
      "expected_objects": [
        { "label": "person", "bbox": [150, 80, 320, 420] }
      ],
      "expected_relationship": null,
      "should_alert": false,
      "notes": "Plain rectangle standing in for a person silhouette. Real detector will very likely NOT match this — that's expected and fine; this entry exists to confirm eval_detection.py runs end-to-end without crashing."
    },
    {
      "image": "_synthetic_smoketest/smoketest_2.jpg",
      "environment": "office",
      "expected_objects": [
        { "label": "laptop", "bbox": [250, 150, 420, 260] }
      ],
      "expected_relationship": null,
      "should_alert": false,
      "notes": "Same purpose as smoketest_1 — harness plumbing check only."
    }
  ]
}