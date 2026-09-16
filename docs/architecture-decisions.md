# Architecture Decisions

## Backbone Strategy: Monocular Depth Estimation
**Decision:** Fine-tune Depth Anything V2 encoder + DPT head on the GAMUS dataset.

**Reasoning:**
- **Leverages Foundation Model**: Depth Anything V2 provides a state-of-the-art pretrained encoder with strong general depth cues.
- **Domain Adaptation**: Fine-tuning on GAMUS (remote-sensing specific RGB-nDSM pairs) addresses the domain gap between egocentric (ground-level) and nadir (satellite) imagery.
- **Hackathon Feasibility**: This approach is the recommended default for hackathon timelines as it avoids training from scratch while providing high accuracy.
- **Balance**: It strikes a better balance between implementation effort and expected accuracy than zero-shot inference or implementing a remote-sensing-specific model from scratch (like HTC-DC Net).

**Expected Outcome:**
Strong performance across diverse landscapes (urban, sparse, hilly, forested) by adapting a general-purpose depth model to the specific geometry of aerial imagery.
