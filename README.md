# List of artifacts

The material collected refers to the IMC23 submission no. 132. Specifically


* code_artifacts_paper132.tgz: All code used for the submission.

* website_documentation.tgz is a website created to 

    1. Gather and document the collection of jupyter notebooks 
        that generates the results of the submission.

    2. Document the curation process followed to prepare
        the datasets for modeling.

    3. Document the modeling framework related to the submission
        (namely tcbench).

* curated_datasets_<dataset-name>.tgz: The preprocessed version of the datasets used in the paper.

* ml_artifacts.tgz: All output data generated via modeling campaigns. 

* unpack_scripts.tgz: A collection of bash script to 
	easy installation of all mentioned artifacts.

__NOTE__: Please refer to the tarballs NOT marked with prefix DEPRECATED.
Those files are left just to show progress in releasing
documentation but they do not reflect the final aim of our artifacts.

# Install artifacts

We recommend to create a specific python environment for the material.
For instance, using conda run

```
conda create -n tcbench python=3.10 pip
conda activate tcbench
```

Download all tarballs in the same folder, and open
`unpack_scripts.tgz`. This will add the following scripts
in the current folder

```
unpack_all.sh
_unpack_code_artifacts_paper132.sh
_unpack_curated_datasets.sh
_unpack_ml_artifacts.sh
_unpack_website_documentation.sh
```

Then run 
```
bash unpack_all.sh
```

This will unpack all tarballs and install then in
the expected position.

# Artifacts exploration

We highly recommend you to explore all artifacts
material via the ad-hoc website.

Using the unpacking described above, this
will be located in the `/site` folder.

Otherwise you can unpack `website_documentation.tgz` manually.

To serve the website locally 
```
cd site
bash ./_serve_site.sh
```


---

Changelog:

* 2023/05/27: uploaded first version of website and code artifacts.

* 2023/06/18: 
    * the website describes the datasets curation process
    * the website integrated the notebooks used for the submission
    * we uploaded ucdavis-icdm19 curated data
    * we uploaded all ml artifacts

* 2023/07/08: 
	* uploaded all pending curated datasets
	* integrated documentation of ML runs and campaigns
	* added script to unpack artifacts

* 2023/07/16:
    * integrated documentation for campaign reports
    * integrated documentation for tcbench internal modules

Pending:
* ~~description of ml campaigns~~ done
* ~~upload remaining curated datasets~~ done
* ~~description to create campaign summary reports~~ done
* ~~improve description of tcbench code structure~~ done
