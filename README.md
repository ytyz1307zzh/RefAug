# RefAug

This is the code repo for the paper *Learn Beyond The Answer: Training Language Models with Reflection for Mathematical Reasoning*. Our work introduces reflective augmentation, a novel technique that aims at cultivating a deeper understanding of the training problems, so as to enhance performance not only in the standard single-round QA settings but also in more complex scenarios that require reflective augmentation. Please refer to our [paper](http://arxiv.org/abs/2406.12050) for more details!

### Environment

Our models are tested on A100 nodes with CUDA version 11.7 and Python 3.9. Please refer to `requirements.txt` for the Python environment we used.

### Data

- The original training data from GSM8k and MATH: `data/original/train.json`.
- The test data for standard math reasoning tasks: `data/original/test.json`. If you are training RefAug model, please make a copy of this `test.json` into the `RefAug` directory. The original unprocessed data from MathInstruct are in `data/MathInstruct`.
- The training data for RefAug: `data/RefAug/train.json`. This is generated using the script `src/data/get_reflection_openai.py`. If you want to use an open-source model to generate the RefAug data, check `src/data/get_reflection_hf.py`.
- The test data for MathChat tasks: `data/original/test-multiturn-followup.json' and `data/original/test-multiturn-error-correct.json`.
- The test data for the math subset of MINT: `data/original/test-mint-original-prompt.json`

### Model

- **Training**: please refer to `scripts/train.sh`
- **Inference**: the training script combines inference, and there is also a separate script called `scripts/inference.sh`
- After inference, for RefAug models, remove the generated reflective section using `src/evaluate/remove_followup.py`. Another option is to add `Reflection:` as a termination string into model decoding.

### Evaluation

- For evaluating **standard math reasoning** tasks, we largely follow the implementation of MathInstruct. Please check `src/evaluate/eval_mathinstruct.py`. Note that for MMLU and SAT, since they are multiple-choice tasks, please first extract the predicted option using `src/evaluate/gpt_extract_answer.py` before calling `eval_mathinstruct.py`.
- For evaluating **MathChat** tasks, check `src/evaluate/eval_multiturn_gsm.py` for follow-up QA and `src/evaluate/eval_error_correction_gsm.py` for error correction. After inference, re-run the script with the same arguments will skip inference and directly show the results.
- For evaluating **MINT**, check `src/evaluate/mint/eval_mint.py`. After inference, re-run the script with the same arguments will skip inference and directly show the results.
- For **error analysis** on GSM8k test set, check `src/evaluate/error_analysis.py`.
- For **contamination test** on GSM8k and MATH, check `src/evaluate/check_overlap.py`

### Citation

### Citation

If you find our data or code useful, please kindly cite our paper:
```
@article{zhang2024refaug,
  title={Learn Beyond The Answer: Training Language Models with Reflection for Mathematical Reasoning},
  author={Zhang, Zhihan and Liang, Zhenwen and Yu, Wenhao and Yu, Dian and Jia, Mengzhao and Yu, Dong and Jiang, Meng},
  journal={ArXiv preprint},
  volume={2406.12050},
  year={2024}
}
```
