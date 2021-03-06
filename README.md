# NLP_imageCaption

an image caption model based on Seq2Seq

## requirements

To run this model, you need put dataset in `media\ssd\caption data` path and its corresponding caption json file in `caption data` path. 
This model supports three datasets: `COCO`, `flickr8k`, and `flickr30k`. 

flicker8k datasets download links: https://pan.baidu.com/s/1q77r2KtxMzE74WNBkaAigw  extract code: `1rh2`

corresponding caption json download links: https://pan.baidu.com/s/1tNAyFucFT0FJw1ebnAItuA extract code: `bcf3`

For more details, you can read these blogs:

https://blog.csdn.net/weixin_44826203/article/details/107609852

https://blog.csdn.net/qq_45893319/article/details/120047448

## some innovation points:

1. change LSTM architecture to modified LSTM architecture
2. N-gram attention: add sequence information to attention mechanism
3. add supervisory signal to encoder and decoder at the same time

## baseline:

https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
