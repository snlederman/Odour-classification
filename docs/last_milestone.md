# Final Project: Last Milestone
<!-- Format: (Submit in PDF/Word/Google Docs format, and include your team number or name in the filename) -->

**TEAM 3**

## List of team members & tasks
<!-- Please list the names of your team members and what each member did, on a high level (up to 2 sentences each) -->

Ella Jewison:
Model testing, medium article writing, testing advance dimensionality reduction techniques and feature extraction...

Ben Sivan:
Model testing, structuring the project for robust model testing and automatic metric logging.

Samuel Lederman:
Researching advance data augmentation techniques.

Ethan Ben-Attar:


## First section: Project name, description & one-paragraph description of
<!-- achievements. Describe very briefly what the current state of your project is,
including one or two metrics that show your performance on the dataset.
Also give a few sentences summarizing the business use of your project. -->

Odor classification:
Classification of odor's signals that where recorded using the desert locust antenna.
Labels - 8 different odors
Features - 150 time points of electical potential measured from the antenna's neurons submitted to various odors.

The dataset is mostly balanced, so our metric of choice for high level evaluation of the model was accuracy. Since we have 8 labels, base model of random sampler would have accuracy of about 20%. In the original paper published, the researcher used random forest model and achived accuracy of 69.3%. After testing multiple strategies for preprocessing and multiple models we also found the random forest gives the best results, but we were able to increase the accuracy to 80%.

## Second section - main part: Technical description of your project. Describe
<!-- the dataset(s) you used, steps you took to analyze and preprocess the data,
and different models you tried. It should include a comparison of your
baseline model with other models that you have tried. Remember to include
a table of results with the format described in MS2. -->

<!-- - This part should be accompanied by a medium article draft (ready for
review), describing all of the above. Iclude visualizations and make the
story appealing - this will be published and you could use it for your
brand. -->

<!-- - Also, prepare a draft for a poster to show in graduation. It should have
visualizations, and a summary of your work. -->

## Third section: Describe the 3 largest challenges that the team had to
<!-- overcome to build this (especially unexpected challenges). -->
Data augmentation:
Dimensionality reduction:
Feature extraction

## Fourth section: What are the remaining steps to complete your project? This
<!-- might include more data modeling, deployment, and preparing the
presentation and write-up. Include how you will divide the work among the
team, with names and who plans to do what.
Lastly, share your git repository. It is recommended to use the DS coockiecutter if
you haven’t yet, create a super organized repository, including a README.md file
which will function as the “cover” for your project. -->

Repository: https://github.com/EllaJewison/Final_project.git