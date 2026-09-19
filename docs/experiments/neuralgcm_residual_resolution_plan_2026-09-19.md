# NeuralGCM residual experiment plan — superseded

The maintained specification is now the
[NeuralGCM residual pipeline README](neuralgcm_residual/README.md).

The user clarified that **four** means the combinations of graph width
**128/256** and Mamba inner width **16/32**, not four graph message-passing steps.
The initial study uses **2.8° and 1.4°**, giving eight configurations, each with
cached **k=1** pretraining and live **k=20** closed-loop fine-tuning.

The README supersedes this document's earlier four-message-pass assumption and
its proposal to select one architecture before a three-resolution sweep. It
retains the existing two-message-pass depth as a proposed default, documents
Mamba carry across BPTT chunks, and distinguishes implemented GC utilities from
the NeuralGCM integration and launcher that still need to be built.
