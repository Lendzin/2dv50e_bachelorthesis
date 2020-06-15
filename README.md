# [2dv50e_bachelorthesis] Bachelor Thesis by: Jonas Strandqvist

* The code belonging to this project is not "clean" in the sense of full understanding, 
functionality was included as the project went along and could thus be refactored quite a bit. 
(also stated in the thesis: most likely better off re-implemented with purely python, 
alternatively fixing issue with javascript and thus having all code in that environment).

* Main function, or "the start of the program" as a client connects is: 'index.js' (in the router folder).

* Models and all files produced as results from these models are stored in : "public/models"
(only models used in this thesis are included, prelimenary models are more than 1GB more of data)

* These files cannot be downloaded and directly run on your local machine, there will be enviromental files as well as package-files missing.
The code here represents the code used for the thesis, but does not contain all un-necessary bits to get it "running".

* All functionality, as well as functionality used to produce prelimenary results are included.
This means that there are functions that would not be used while producing the results for the actual thesis, also included amongst these.

* 'Unused' outcommented functionality have been removed to clean the project up, however should not mess with actual functionality.
Some outcommented things have remained as they might be useful for anyone who wants to play around with functionality related to said functions.

* Converting models from JavaScript to Python was done by using the following command:

<div align="center">
tensorflowjs_converter \ <br>
 --input_format=tfjs_layers_model \<br>
 --output_format=keras \<br>
 path_to_javascript_model/model.json \<br>
 \name_of_pythonmodel<br>
</div>
