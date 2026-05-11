1. I use the uv package manager, with python 3.14. 
2. All the code must be stored in src/nlp_news and scripts/
3. Review the following requirements and provide a sensible critical opinion about them, before proposing an implementation plan.
4. I want a pipeline with the following stages:
   3.1 A python script that reads text files (in a directory specified in a config yaml file), divide them in words, generates n-grams (n specified in the config yaml file) and stored them in a sqlite database.
   3.2 A python script that finds the non boring n-grams and use them as items in itemsets to create a table of itemsets in the sqlite database
   3.3 A python script that finds association rules in the table of itemsets.
   3.4 A web server that shows a visualization of the association rules and the text files.
   
   
I want to make a python script to import the html and xml files in @data/samples for processing with the pipeline. Do not try to review all these files with your tools: read no more than three samples. Make a proposal for review before generating any code.   
