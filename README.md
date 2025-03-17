# Take Home Assignment

## About this repository
* This repository contains a solution to the HopSkipDrive Take Home Assignment for the Senior Data Scientist Role. 
* The repository is public, as requested in the assignment instructions. 
* The repository contains the code to produce the following three deliverables for this assignment:
	* `written_report/takehome_notebook.html` is the rendered version of the written report in HTML format. 
	* `takehome_notebook.ipynb` is the Jupyter Notebook I used to develop the model and generate the written report.
	* `model_pipeline.py` is the Python script used to simulate a script running in production. The script also generates `data/model_df.csv` when executed, as requested in the instructions. That file contains the data ready for modeling. As requested in the instructions, this repo does not include any data so it is necessary to run the script to produce `data/model_df.csv`.

## Prerequisites
* Both `model_pipeline.py` and `takehome_notebook.ipynb`  assume there is a directory called `data/` at the same directory level as the notebook and a file called `boost_df.csv` inside that directory.

## Running the model
* Using Docker, start by building the image with `docker build -t elastic_net_pipeline .` and then run the Python script inside the container with `docker run --rm -v $(PWD)/data:/app/data/ -it elastic_net_pipeline python3 model_pipeline.py`. Note that the performance of the model and coefficients are displayed in the terminal as part of the logging when running the script. Running the script also outputs the file `data/model_df.csv` outside of the container. 
* You can also create a virtual environment and install the `requirements.txt` to run the script without using Docker.

## To render the written report:
*  Similar to running the model, you can run render the notebook in Docker using `docker run --rm -v $(PWD):/app -w /app -it elastic_net_pipeline quarto render takehome_notebook.ipynb --execute --output-dir /app/written_report/`
* Or you can directly render it in your virtual environment after installing requirements: `quarto render takehome_notebook.ipynb --execute --output-dir written_report/`. 