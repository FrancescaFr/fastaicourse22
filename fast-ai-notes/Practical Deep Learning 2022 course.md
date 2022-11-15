---
title: Practical Deep Learning 2022 course
updated: 2022-11-15 05:42:43Z
created: 2022-07-30 17:34:36Z
latitude: 45.52306220
longitude: -122.67648160
altitude: 0.0000
---

# Summary
These are my notes for [Practical Deep Learning for Coders 2022](https://course.fast.ai/) a free online course taught by Jeremy Howard - which is structured as a series of Jupyter notebook assignments and recorded lectures.

This course is complimented by a physical book [Deep Learning for Coders with Fastai and PyTorch: AI Applications Without a PhD](https://course.fast.ai/Resources/book.html) 2020 Print Edition. There is also an Online Edition [on github](https://github.com/fastai/fastbook/blob/master/01_intro.ipynb) with its own Jupyter Notebook assignments. 

The fact that there are two extremely similar, yet different sets of "fast.ai" course materials can be confusing, but to be clear, I will be  watching the lectures and completing the  ***notebook assignments*** from the newest course 2022 fast.ai course, as they are the most current. I will also read the 2020 book - skipping the assignments and focusing on the more detailed explanation of ML concepts and approaches. 

*italicize terms or ideas are own additions/asides*

# Lesson 1 Lecture Notes: Getting Started

[Lesson 1 Video](https://www.youtube.com/watch?v=8SF_h3xF3cE)
[Lesson 1 Forum Post](https://forums.fast.ai/t/lesson-1-official-topic/95287)

## Topic 1: Is it a bird? - Image Classifier
Jumps straight into example CV, does not touch theory initially 

### Application Topics:
- notes deep learning + art custom model design
- Language Model (Google Pathways Language Model PaLM explainer)

### Companion reading:
- **Recommends reading corresponding 2020 book chapter after each video**
- Practical Data Ethics ([ethics.fast.ai](/C:/Program%20Files/Joplin/resources/app.asar/ethics.fast.ai "ethics.fast.ai")) \- Data Ethics Course 
- A Mathematician's Lament, Paul Lockhart (Book Rec)

### Tools Used in Lecture:
- [Student Confidence tracker](https://cups.fast.ai/)
- [Pixspy](https://pixspy.com/) - allows you to read RGB values of image regions
- Jupyter notebook extension: [rise.readthedocs.io](/C:/Program%20Files/Joplin/resources/app.asar/rise.readthedocs.io "rise.readthedocs.io") turns notebook into presentation...or books, or blogs, or tests, or source code.
- **gv2 graph vis function in python for creating visuals**


## General Notes
- **I like that they give you a little history on feature recognition, and how NN is better, but doesn't emphasize what NN is vs previous Machine Learning tools**
	- *Topics they're mentioning, but not explaining yet: features, weights, gradients, layers NN, Deep Learning*

- Think creatively about data: 	
	- **An Image classifier can serve as a sound classifier, time series, etc. by creating images of waveform, time, movement** - you can use a model on lots of data if you are creative about formatting the data before training on it.
	- You don't need tons of data or expensive computers to train usable models - if you are starting with pretrained especially
- PyTorch is used throughout this course. It is more popular than TensorFlow generally because there are more libraries/research/community/ support around it
	- *What are libraries? Pytorch, etc.* - why are they important?
	- *fast.ai library is built on top of PyTorch*  - it constrains you to best practices and is simpler to use

## **Introduces Jupyter Notebook**

- gives you different cloud server options (Gradient, Kaggle, SageMaker, CoLab)
	- *should we recommend a specific server option? likely google colab. Maybe Paperspace Gradient?*
	- *we should show people how to use their local machine with google colab?*
	- *Most students will need a more detailed jupyter notebook introduction before this course, along with common bash shell commands , ex.!pwd*

## Machine Learning Model Workflow
- Models are largely figured out for most applications...so you don't often have to start from scratch
- DataBlock : data preparation is a very important
    - Ask: what are the critical features?
        - input type (image), output (category) --> determines category for model
        - retrieval of inputs (parent folders)
        - splitter (*validation set*)
        - labels (identifying set labels)
        - data manipulation (resizing, etc.)
        - `.dataloaders(path)`<-- GPU function loading batches of data into model
        - Reference: **[docs.fast.ai](/C:/Program%20Files/Joplin/resources/app.asar/docs.fast.ai "docs.fast.ai") gives you fast.ai api info (terms, functions etc.)**
        - timm.fast.ai Pytorch Image Models is integrated with fast.ai
- Learning (requires data, model type, metrics)
- Models are often pre-trained (ex. on imagenet), and by default fast.ai starts with pre-trained weights to give it a head start, then permits fine tuning to specific application.
- After training, you can use your model for prediction with a single line of code

## Other forms of ML

### Segmentation

- Segmentation - classification of objects/regions inside images
- Data loader Classes for specific inputs (as opposed to datablocks)
	- *starting to want more information here - what kind of data does this class expect? what exactly does it do vs data blocks?*

### Tabular Analysis

- `untar_dat` (downloads and decompresses data sets)
- Tabular data loader class
- tabular models **do not usually use pre-trained models,** require new model 'fit'

### Collaborative filtering

(recommendation systems)

- find "similar users" by interests, rather than demographics
- need to define an output range (0.5,5.5) (use slightly off of actual range)
- "mean spread error" 
	-*could discuss intro statistics a bit here*

## What can Deep Learning do?

NLP, CV, Medicine, Biology, recommendation systems, playing games, robotics...

-"When I say these are things that it's currently state of the art for - these are the ones that people have tried so far - but still, most things haven't been tried" - JH
- Mark 1 Perceptron at the Cornell Aeronautical Laboratory (1957)  was the first Neural Network
- The big difference now versus when some of these ideas first emerged: the amount of data, size of SSDs and GPUs.
- Arthur Samuel - Definition of ML (*missed*)
- A model is a mathematical function, rather than a 'program' of conditionals and loops
	- Needs weights and inputs to generate results, those results generate a 'loss' which feeds back into weights (update weights based on loss, iterating until loss is minimized)
	- Model must be flexible enough to adapt to new weights (NN is infinitely flexible - **computable function**)
	- Once training is complete, loss isn't required, weights can be integrated into model, and it looks like original idea of a program, which can be called with inputs to generate outputs

## Getting started...

Folks without Python and Jupyter Notebook knowledge will have a harder time (seek external resources).

There is a forum on the website as well as a Discord channel (lots of community support)

1.  Run Kaggle notebooks (or Colab)
	- Then, tweak the notebooks
	- Then, build your own notebook
4.  Read Chapter 1, do Quiz questions
5.  Share what you did with your peers (*we can do in our group*)
6.  look around see what other examples there are for applying your type of model (disaster GIS, envision, fraud detection, etc.)


# Notebook 1 Notes: Classifier
The original class example involved training a "dog or cat" classifier. The instructor was able to achieve quite impressive results rather quickly. I opted to build a classifier for "Crocodile or Alligator" instead. 
My thinking was:
1. Cat's and Dogs are pretty different, it might be more useful to distinguish between animals that humans get confused.
2. How well does this naive image scrapping technique hold up for more niche subjects? 
3. What flaws or relevant parameters in this training approach might be revealed by more polluted data set?

## Attempt Take 1:

- Had some errors when importing images, much dirtier data set:
    ![51437202ce1f140c230a71d5064cf29e.png](../../../_resources/51437202ce1f140c230a71d5064cf29e.png)
- Error rates are much higher: increased fine tuning epochs to decrease error rate:
    ![38ca0fbc8199d48f6f13c3d77b6321ec.png](../../../_resources/38ca0fbc8199d48f6f13c3d77b6321ec.png)
- Results: Min error: 15% after 6 epochs. Confident alligators are alligators, also fairly confident crocodiles are alligators :/
    ![43c748c3776d62df3d0e5c12ebf955ac.png](../../../_resources/43c748c3776d62df3d0e5c12ebf955ac.png)
    ![b690104d8c298a01a479f40f1e651abe.png](../../../_resources/b690104d8c298a01a479f40f1e651abe.png)

## Attempt 2:

- Realized, though i eliminated "photo" from search for the first results, I didn't eliminate the term "photo" for the batch image search - the photo term resulted in a bunch of non-animal images (logos, shoes and bags, etc)
- After eliminating, error dropped to 12% after 4 epochs and was able to correctly identify initial crocodile vs alligator images:
    ![e1f2e382710452b426273b2b984c88e4.png](../../../_resources/e1f2e382710452b426273b2b984c88e4.png)

Interesting example of need for more curated data set!
Can I trust people are correctly uploading/distinguishing between crocs/gators? 
Would I need an expert to comb through and curate a data set for me?

# Additional Notes

## Ancillary lectures and support materials
Current Topic: [Programming in APL and Array Programming](https://forums.fast.ai/t/apl-array-programming/97188)
5PM Mon/Thurs PST
[Live Coding Video Playlist](https://www.youtube.com/playlist?list=PLfYUBJiXbdtSLBPJ1GMx-sQWf6iNhb8mM)

Recordings starting simple...gets more advanced
[Session 1: Command Line, Python, PyTorch, Jupyter](https://forums.fast.ai/t/live-coding-1/96649)
[Session 2: Github, SSH, Tmux, fastai](https://forums.fast.ai/t/live-coding-2/96690)
[Session 3: %PATH environment variable, conda environemnt, paperspace notebook, pip, Symlinks, storage, etc](https://forums.fast.ai/t/live-coding-3/96707)

## *HOW TO: Bringing Jupyter Notebooks over to Paperspace Gradient*

https://console.paperspace.com

These are the steps I took to bring in and get the notebooks running in Paperspace gradient - for GoogleColab, **note step 5**.
1.  Create account (associated with github)
2.  Create a project with a PyTorch 1.12 Runtime (NOT fast.ai)
3.  Open and start machine
4.  Go to JupyterLab > Command Line > git clone fast.ai Course 2022 repo (I forked my own and then cloned)
5.  **Remove references to kaggle (ex. first cell in first notebook only imports fastai library if in kaggle, remove that conditional and import the library)**
6.  So far so good...
7.  pushing to my forked version on github:
    1.  [Need a github personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token) Github settings > developer settings > created 90 day exp. token with repo access for paperspace.

A paperspace primer that i didn't read but might be helpful: https://blog.paperspace.com/gradient-community-notebook-guide/

# Lesson 2: Model to Production
[Lesson 2 Video](https://www.youtube.com/watch?v=F4tvM4Vb3A0)
[Lesson 2 Forum Post](https://forums.fast.ai/t/lesson-2-official-topic/96033)
Jupyter Lab extensions are available (nbextensions) that can be helpful for navigation

**bing api is now replaced with ddg bc it doesn't require a key (`search_images_ddg`)**

Double questionmark next to function name gives you source code (??verify_images). Single gives you brief explanation 

Resizing options: (Squish, crop, pad, random crop- different pics per epoch (data augmentation))

another augmentation is aug_transform, which slightly deforms/tilts images each time (real time during training, not making copies of files)

1. train a model
2. run ClassificationInterpretation function.
	Look for Category errors: Confusion matrix (shows what classifications were mistaken for what),
	can run top_loss function to find what were the most "wrong " guesses. 
3. clean your data
- run ImageClassifierCleaner(model) to remove incorrect or missattributed items.

## Deployment
[4. Gradio + HuggingFace Spaces: A Tutorial](https://tmabraham.github.io/blog/gradio_hf_spaces_tutorial)
(No chapter, a blogpost)

Go to [huggingface.co/spaces](huggingface.co/spaces)
- Creates a space under git

**- Running a full Linux environment in Windows!**
	- [Install WSL](https://docs.microsoft.com/en-us/windows/wsl/install)
		- Installs Ubuntu by default
		- [wsl setup best practices](https://docs.microsoft.com/en-us/windows/wsl/setup/environment#set-up-your-linux-username-and-password)
			- tips like: how to reset password, access in file explorer, find root directory, git, docker, GPU acceleration

### Install VSCode for WSL
[https://docs.microsoft.com/en-us/windows/wsl/tutorials/wsl-vscode](https://docs.microsoft.com/en-us/windows/wsl/tutorials/wsl-vscode)
1. Install VS Code, python and the Remote WSL extension
	- possible useful guide:  [Get started using Git on Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/tutorials/wsl-git)
Visit the VS Code install page and select the 32 or 64 bit installer. Install Visual Studio Code on Windows (not in your WSL file system).

2. When prompted to Select Additional Tasks during installation, be sure to check the Add to PATH option so you can easily open a folder in WSL using the code command.

3. Install the Remote Development extension pack. This extension pack includes the Remote - WSL extension, in addition to the Remote - SSH, and Remote - Containers extensions, enabling you to open any folder in a container, on a remote machine, or in WSL.

- *Note: At about 30 minutes in, Jeremy drops in Git (command line, etc) with minimal context, we should have some intro to git before this part in the video*
	- Also, instruct users to navigate to specific location to save github repos
	- need to create a file: 
		1. open repo in vscode (jeremy recommends *vscode - note: add to list of recommended programs, maybe explain what IDE is*)
		2. create app.py with provided code
		3. commit and push (with huggingface credentials)
		4. Now you can go back to huggingface space to see your app 

*Note from Hugging face site:
Dependencies:
You can add a requirements.txt file at the root of the repository to specify Python dependencies
If needed, you can add also add a packages.txt file at the root of the repository to specify [Debian dependencies](https://huggingface.co/docs/hub/spaces-dependencies).
The gradio package comes pre-installed at version latest
Documentation:
Check out the documentation of gradio on their [website](https://gradio.app/docs)*

5. Lesson two - grab a model from Lesson 2 Forum Post
[https://www.kaggle.com/code/jhoward/saving-a-basic-fastai-model/notebook](https://www.kaggle.com/code/jhoward/saving-a-basic-fastai-model/notebook)
`learn.export('model.pkl')`
Download and paste into huggingface space repo (?) not clear

*note: he starts talking about your "linux machine and saving files in there", but he never actually explains where your linux os or its files live, or why/that you have to save previous files there, including the github repo*

For me, this is where my linux files are saved: `\\wsl$\Ubuntu\home\francescafr`
Navigate to this folder from ubuntu terminal (`explorer.exe .`)

*it's getting unclear here where he the files are that he is opening, it appears he is opening a local jupyter file to test the model, but that is not mentioned in the video*

- Any external functions used in labeling needs to be included in new file (meaning, you need to reference label terms as defined in the model, so you need to keep track of source code or have list of labels with your model pkl)
- load_learner('model.pkl')
- learn.predict(image)

Gradio requires a function (call learnpredict) and a dictionary of possible categories.
Gradio doesn't handle tensors(*what are tensors, you might ask?*)


### Exporting from notebook to app file:

include `!|export` in each cell of jupyter notebook that you want as part of final script.
`from nbdev.export import notebook2script`
then you can use function to generate app.py `notebook2script('app.ipynb')` 
and it will save it in the same folder. Then you can run it locally (*need to show how*)
