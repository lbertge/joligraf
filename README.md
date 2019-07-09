# Joligraf

Joligraf is a project that aims to regroup high-level data visualization graph as TSNE / UMAP interactive projection, or confusion matrix, etc...

## Motivations

1. I was repeating myself when it comes to create visualisation. Matplotlib or other visualization libraries are great tools but I was redefining the confusion matrix or other graph each time I needed one.
2. For some plot as TSNE or UMAP projection, I need it more interactive and be able to find the image source I looking at.

## How to use it
### Installation
This repo uses `poetry` as dependencies manager.

```
poetry install
```

### Usage


```
python joligraf/cli/video/videos_extract_frames.py ~/data/MineRLTreechop-v0/r2g1absolute_grape_changeling-15_12958-15286/
python joligraf/cli/rl/sequence_pair_img_act.py ~/data/MineRLTreechop-v0/r2g1absolute_grape_changeling-15_12958-15286/ --start=0 --limit=50 --out=holo4.html
```
