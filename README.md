# roci


Toolbox for detecting and removing RFI from Green Bank Telescope data.


Will include many different algorithms.






### Getting roci for yourself


`git clone git@github.com:etsmit/roci.git`


### Development

#### Github etiquette

Start from the main branch, which should be the most up-to-date:

`git fetch`

`git checkout main`

`git pull origin main`

Create a new branch named after an RFI algorithm you want to add

`git branch new-branch-name`

Make any edits/additions. When you want to commit and push to Github,

`git add .`

`git commit -m "commit message"`

`git push origin new-branch-name`

And when you want to add your code back to main, open a pull request on Github.

#### Adding your code


Create a copy of mitigateRFI_template.py and rename the "_template" to an acronym or initialism pertaining to your RFI algorithm, such as "IQRM" for inter-quartile range mitigation. You'll replace the section starting at line 220 with a function that calls your code. The instructions are right there in the code. Add the RFI detection function(s) to RFI_detection.py. Don't forget to add parameters for the RFI detection algorithm and any intermediate numpy arrays you want to generate.



### Using roci

Only need to do this once: Create a conda environment and install the required packages:

`conda create --name [env-name] python=3.9`

`conda activate env-name`

`pip install -r requirements.txt`

Run the code with python:

`python mitigateRFI_IQRM.py -i [input filename] -r [replacement strategy] [-any other parameters]`

utils.py:template_parse() has more arguments in addition to any that you add.







