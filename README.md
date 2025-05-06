# TimeChat-Online: 80% Visual Tokens are Naturally Redundant in Streaming Videos</h1>

<p align="left">
🔗 <a href="https://timechat-online.github.io/" target="_blank">Project Page</a> · 📖 <a href="https://arxiv.org/abs/2504.17343" target="_blank">Paper</a> · ⭐ <a href="https://github.com/yaolinli/TimeChat-Online" target="_blank">GitHub</a> · 📊 <a href="https://huggingface.co/datasets/yaolily/TimeChat-Online-139K" target="_blank">Dataset</a> · 🤗 <a href="https://huggingface.co/wyccccc/TimeChatOnline-7B" target="_blank">Checkpoints</a>
</p>

📰 **News**

- **[2025-05-01]** Our paper and project page are now available.

🚀 **Coming Soon**

- [ ] Model checkpoints and inference code
- [ ] Full training code, scripts, and benchmark evaluation tools

## Introduction



**TimeChat-Online** is a novel online VideoLLM designed for efficient streaming video understanding. Its core innovation, the **Differential Token Drop (DTD)** module, tackles visual redundancy by selectively preserving only meaningful temporal changes while eliminating static content between frames. Our experiments show that over 80% of streaming video content is naturally redundant without requiring user-query guidance.


<img width="1073" alt="image" src="https://github.com/user-attachments/assets/b9ad1d0b-10e0-4125-8a06-2216eef1fcc3" />


## Key Features of TimeChat-Online

✨ **Video-aware Dynamic Pruning:**
DTD adaptively reduces video tokens from a holistic video perspective, well-suited for both high-speed and slow-motion videos.

📝 **Positional Reservation:**
DTD maintains the fine-grained spatial-temporal positions of retained tokens via M-ROPE, ensuring precise spatial localization and temporal understanding capabilities.

⚡ **Streaming-friendly Design:**
DTD efficiently processes video streams by calculating redundancy only for newly-arriving frames, without re-processing historical video content.

🎉 **Proactive Response Capability:**
TimeChat-Online naturally monitors video scene transitions through the DTD module, enabling autonomous identification of critical moments that require responses without explicit queries.


<img width="967" alt="image" src="https://github.com/user-attachments/assets/2b6c65b0-f165-4ce7-841d-4cd8905998aa" />


## Model Architecture
<p align="center">
<img src="https://timechat-online.github.io/static/images/model.png" alt="TimeChat-Online Model Architecture" style="width: 100%;">
</p>


## Performance Highlights

- 🏆 **StreamingBench**: Achieves **56.6** accuracy with **82.6%** token reduction (new SOTA)
- 🏆 **OVO-Bench**: achieves 45.6 accuracy with 84.8% token dropped (new SOTA)
- 🏆 **Long Video Benchmarks (MLVU, LongVideoBench, VideoMME)**: Up to **85.0%** reduction in video tokens while maintaining or improving performance
- 🏆 **Zero-shot Integration**: When integrated with Qwen2.5-VL-7B without training, DTD improves VideoMME (long subset) accuracy by **5.7** points while reducing **84.6%** of video tokens




## Quick Start
[todo] Coming soon within the next week.


## Dataset: TimeChat-Online-139K

For flexible real-time interaction, we introduce a comprehensive streaming video dataset with backward-tracing, real-time visual perception, and future-responding scenarios.

- **11,043** visually informative videos (average duration: 11.1 minutes)
- **139K** question-answer pairs covering backward tracing, real-time visual perception, and forward active responding
- Average of **87.8** scene-oriented key frames per video (~7.14 seconds between consecutive frames)

[todo] We will release the video frames at 1 fps and the question-answer pairs soon.

## Training
We utilize the ms-swift framework for model training. Please note that the training script requires modifications to both ms-swift and transformers code. For detailed instructions, refer to the guidelines in [`train/readme.md`](./train/readme.md) before execution.


## Citation

If you find our work helpful, please consider citing:
```
@misc{timechatonline,
    title={TimeChat-Online: 80% Visual Tokens are Naturally Redundant in Streaming Videos}, 
    author={Linli Yao and Yicheng Li and Yuancheng Wei and Lei Li and Shuhuai Ren and Yuanxin Liu and Kun Ouyang and Lean Wang and Shicheng Li and Sida Li and Lingpeng Kong and Qi Liu and Yuanxing Zhang and Xu Sun},
    year={2025},
    eprint={2504.17343},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2504.17343}, 
}
```

## License and Acknowledgments

- The data, code, and checkpoints are intended and licensed for research use only
- They are restricted to uses that follow the license agreements of the respective datasets and models used in this work

**We thank the following projects for their contributions:** [TimeChat](https://github.com/RenShuhuai-Andy/TimeChat), [Qwen2.5VL](https://github.com/QwenLM/Qwen2.5-VL), [RLT](https://github.com/rccchoudhury/rlt), [VideoLLM-online](https://showlab.github.io/videollm-online/), [OVOBench](https://github.com/joeleelyf/ovo-bench), [StreamingBench](https://github.com/THUNLP-MT/StreamingBench)
