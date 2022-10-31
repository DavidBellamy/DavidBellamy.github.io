---
layout: post
title: How to make an online avatar with deep learning
date: 2022-10-31 17:15:00
description: A quick tutorial on using Dreambooth to fine-tune stable diffusion
---

**TLDR;** this tutorial is an extension to the instructions from [Xavier Xiao's repo](https://github.com/XavierXiao/Dreambooth-Stable-Diffusion) with some gaps filled and tips added.

**Summary**: Since their release in December 2021, [latent diffusion models (LDMs)](https://arxiv.org/abs/2112.10752) have completely taken over image synthesis. [DreamBooth](https://dreambooth.github.io/), released 2 months ago, allows us to fine-tune an LDM called Stable Diffusion on images of our choice and then generate related images via text prompts. We can use use this approach to fine-tune Stable Diffusion on images of ourselves and then generate beautiful avatars.


## Tutorial Steps:
1. Clone [this repo](https://github.com/XavierXiao/Dreambooth-Stable-Diffusion) to your local development environment.
- If you don't have a local GPU, set up a remote sync with this repo and your GPU environment.
2. Set up a conda environment on your GPU server by running ```conda env create -f environment.yaml```
	- Make sure that the custom taming-transformers and clip installations work; otherwise do them manually.
3. Download the diffusion model weights from Hugging Face [here](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/tree/main). 
	- To download them directly to your GPU environment, do it programmatically with:
    {% highlight python %}
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="CompVis/stable-diffusion-v-1-4-original", 
    filename="sd-v1-4-full-ema.ckpt", use_auth_token='your_hf_auth_token', cache_dir='.')
{% endhighlight %}
    
* Note: beware of the dash between 'v' and '1' in the repo name, some sources don't have it and it will lead to 404 Not Found errors. Also, you can create a Hugging Face authentication token in your Hugging Face profile's settings under Access Tokens.

* The downloaded weights will end up in a directory of the form: ./models--CompVis--stable-diffusion-v-1-4-original/snapshots/0834a76f88354683d3f7ef271cadd28f4757a8cc/sd-v1-4-full-ema.ckpt. Use this path to reference the weights when fine-tuning the diffusion model.

{:start="4"}
4. Gather your training images for fine-tuning the diffusion model.
	- If you want to fine-tune the model on yourself, the recommendation is:
		- 2-3 full body
		- 3-5 upper body
		- 5-12 close-up on face
5. Gather your regularization images, to avoid overfitting.
	- These should be images in the same 'category' as your training images, yet distinct. So if you are finetuning on yourself, gather images of other people. 
	- I recommend gathering these images from the internet rather than generating them from the diffusion model (as the Github repo suggests).
	- The more regularization images you have, the better. 100+ is apparently ideal, but a lot of effort. I got away with just 8.
6. Finetune the diffusion model with this command
   {% highlight python %}
   python main.py --base configs/stable-diffusion/v1-finetune_unfrozen.yaml 
                -t 
                --actual_resume /path/to/sd-v1-4-full-ema.ckpt  
                -n <job name> 
                --gpus 0, 
                --data_root /root/to/training/images 
                --reg_data_root /root/to/regularization/images 
                --class_word <xxx>
   {% endhighlight %}
- For the class_word, I recommend using a noun, not an adjective/style. Ex. do not use "portrait", use "man" or "woman" instead.
- On one A100, the training takes ~15 min or so.
- Note: the gpus '0' argument is not referring to "no GPUs", it refers to the device number.

{:start="7"}
7. Generate samples!
{% highlight python %}
python scripts/stable_txt2img.py --ddim_eta 0.0 
                                 --n_samples 8 
                                 --n_iter 1 
                                 --scale 10.0 
                                 --ddim_steps 100  
                                 --seed 1
                                 --ckpt /path/to/saved/checkpoint/from/training
                                 --prompt "photo of a sks <class>"
{% endhighlight %}

- Your checkpoint from training will be under a directory of the form: `logs/train_images2022-10-31T10-23-15_job1/checkpoints/last.ckpt`
- Note: I added a seed argument so that you can change it if you wish to generate more samples using the same prompt as prior attempts; just change the seed.

## Tips on Prompting
For prompting, I recommend looking through images at [lexica.art](https://lexica.art/) and [this Notion page](https://proximacentaurib.notion.site/e28a4f8d97724f14a784a538b8589e7d?v=ab624266c6a44413b42a6c57a41d828c).
- Find images you like and try their prompts yourself! You will also learn about good prompting structures/keywords, like wlop, loish, hyper realistic, etc. which you can use in prompts of your own design to better effect.
