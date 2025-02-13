# Correlation and Navigation in the Vocabulary Key Representation Space of Language Models [[📃Paper]](https://arxiv.org/abs/2410.02284)

Some knowledge is memorized by LLMs, but you cannot prompt to get it - we provide a simple we to do so. Here's why and how.

![KeyNavi](https://github.com/user-attachments/assets/5dacdfa9-12bc-4431-a10b-6460edbf0bd6)

1) Neural LMs suffer from spurious correlations in next-token prediction (NTP), where mid-ranked tokens are biased toward distributionally similar but incorrect options, reducing sampling diversity.

2) We propose an iterative in-context method (ICN) that updates the query representation by including explored decoding results, pushing the LM away from previously selected tokens.

3) ICN improves generation diversity and self-consistency in knowledge probing, open-ended generation, and chain-of-thought reasoning.

## ⚡ Quick start

### The clusters of vocabulary representations

| CID  | In-Cluster subwords |
|------|---------------------|
| 896  | A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, ... |
| 200  | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 00, 20, 10, 201, ... |
| 640  | Ġwonder, Ġbeautiful, Ġamazing, Ġexcellent, ... |
| 996  | urn, Ġecho, printf, echo, Ġprintf, ĉecho, ... |
| 100  | elf, eld, elp, els, EL, elt, elves, ael, El, elay, ... |
| 350  | lob, Ġlunch, Ġlens, Ġlip, Ġlobby, Ġlaptop, ... |


You can check the illustration of vocabulary representation clusters in Huggingface for [llama-3](https://huggingface.co/datasets/KomeijiForce/llama3_vocabulary_cluster) and [olmo](https://huggingface.co/datasets/KomeijiForce/olmo_vocabulary_cluster). Some cluster examples are illustrated above.

### Spurious Key Correlation

In ```Spurious_Key_Correlation.ipynb```, we provide a pipeline to show how the next tokens predicted from different clusters from the top predictions are more accurate.

## 🐾 Citation 

```
@article{DBLP:journals/corr/abs-2410-02284,
  author       = {Letian Peng and
                  Chenyang An and
                  Jingbo Shang},
  title        = {Correlation and Navigation in the Vocabulary Key Representation Space
                  of Language Models},
  journal      = {CoRR},
  volume       = {abs/2410.02284},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2410.02284},
  doi          = {10.48550/ARXIV.2410.02284},
  eprinttype    = {arXiv},
  eprint       = {2410.02284},
  timestamp    = {Thu, 07 Nov 2024 15:42:46 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2410-02284.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
