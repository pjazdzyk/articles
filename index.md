# Engineering Publications & Technical Reports

---

## Recent article
### Memory-Constrained Quantization Analysis of Qwen3-Coder-30B: Balancing Reasoning Integrity and Extended Context Capacity

> **Abstract:** This technical report investigates the impact of model weight and KV cache quantization on the performance of the Qwen3-Coder-30B model under extended context conditions. Experiments were conducted using a fixed 128 000 token context window on a 32GB consumer-grade GPU. The results reveal several unexpected performance characteristics. In particular, 8-bit KV cache quantization (Q8_0) scores slightly higher than FP16 baseline across both LongBench-v2 and BigCodeBench metrics. Furthermore, Q4_K_M weights demonstrate stronger reasoning integrity than the less aggressive Q5_K_XL quantization. While Q3_K_L remains the only configuration capable of fully supporting a 256k context window, it introduces measurable performance degradation, with the exception of the Q3_K_L / Q8_0 KV combination which performs unexpectedly well in the LongBench “long” category. Overall, the configuration combining Q4_K_M weights with Q8_0 KV cache emerges as a practical sweet spot, enabling an estimated 225 000 token context capacity while maintaining accuracy above the FP16 baseline.

### **[Read the Full Technical Report](2026-02-qwen3-kv-cache-quantization/docs/Jazdzyk2026-Qwen3-Quantization-Analysis.md)**

---

## Archive (PDF)

* **[March 2026]** - [Memory-Constrained Quantization Analysis of Qwen3-Coder-30B: Balancing Reasoning Integrity and Extended Context Capacity](2026-02-qwen3-kv-cache-quantization/docs/Jazdzyk2026-Qwen3-Quantization-Analysis.pdf)