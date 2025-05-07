### Download dataset
download dataset from [TimeChat-Online-139k](https://huggingface.co/datasets/wyccccc/TimeChat-Online-139K),[LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K),[Tarsier2-Recap-585K](https://huggingface.co/datasets/omni-research/Tarsier2-Recap-585K),[VideoChat-Flash-Training-Data](https://huggingface.co/datasets/OpenGVLab/VideoChat-Flash-Training-Data)

Then replace the video paths in the JSONL files from [TimeChat-Online-139k](https://huggingface.co/datasets/wyccccc/TimeChat-Online-139K)

### Create Conda Environment

```bash
conda create --name TimeChatOnline-train python=3.10
conda activate TimeChatOnline-train
````

---

### Set Up `ms-swift` from Source

```bash
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
git checkout v3.2.0
pip install -e .
cd ..
```

---

### Set Up `transformers` from Source

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout v4.49.0
pip install -e .
cd ..
```

---

### Replace Specific Source Code Files

> *Note: The following is a hard-coded override of selected source files. A more flexible solution is coming soon.*

```bash
# Replace files in ms-swift
mv train/ms-swift-replace-code/base.py train/ms-swift/swift/llm/template/base.py
mv train/ms-swift-replace-code/qwen.py train/ms-swift/swift/llm/template/template/qwen.py

# Replace file in transformers
mv train/transformers-replace-code/modeling_qwen2_5_vl.py train/qwen_2_5/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py
```

---

### Launch Training Script

```bash
bash train/train.sh
```
