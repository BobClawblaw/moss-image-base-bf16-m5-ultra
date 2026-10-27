# Gallery

Five pictures made by this recipe's `python3 -m mac generate` on one M5 Ultra from checkpoint
revision 85975793, at the card's defaults (50 steps, guidance 3, the empty negative prompt skipping block 22). Nothing was
picked or retouched: each is the one picture its prompt and seed gave.

They were made on engine 0b69e127, and made again on the pinned 772283b6 with byte-identical results. The times are
the second run's, with no other GPU work.

| picture | size | seed | transformer |
|---|---|---|---|
| [1-bookshop.png](1-bookshop.png) | 1024 x 1024 | 42 | 47.2 s |
| [2-portrait.png](2-portrait.png) | 1024 x 1024 | 7 | 47.3 s |
| [3-sign.png](3-sign.png) | 1024 x 1024 | 42 | 48.6 s |
| [4-fridge-zh.png](4-fridge-zh.png) | 1200 x 880 | 7 | 48.8 s |
| [5-valley.png](5-valley.png) | 1344 x 768 | 3 | 45.7 s |

`4-fridge-zh.png` is byte for byte the picture of the `fridge-zh-1200x880-7` check, made in an earlier session.

## Prompts

1. A cozy bookshop interior on a rainy evening, warm lamplight, a cat sleeping on a stack of old books, rain streaking
   the window
2. Close-up portrait of an elderly fisherman with a weathered face and a knitted wool cap, soft overcast light, shallow
   depth of field, photograph
3. A vintage enamel shop sign that reads "OPEN 24 HOURS" in bold red letters, mounted on a brick wall
4. The model card's prompt, [cases/fridge-zh.txt](../../cases/fridge-zh.txt)
5. A wide mountain valley at sunrise, mist over a winding river, pine forests, dramatic clouds, landscape photograph

![bookshop](1-bookshop.png)
![portrait](2-portrait.png)
![sign](3-sign.png)
![sticky note](4-fridge-zh.png)
![valley](5-valley.png)
