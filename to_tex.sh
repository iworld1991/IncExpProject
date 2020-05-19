#!/bin/bash

cd WorkingFolder/PythonCode/


ipython PerceivedIncomeRisk.py
jupyter nbconvert --output-dir='./latex/' --to=latex --LatexExporter.template_file=./latex/PerceivedIncomeRisk0.tplx TagRemovePreprocessor.remove_cell_tags='{"remove_cell"}' PerceivedIncomeRisk.ipynb

ipython TexTablesMover.py

cd ./latex/

pdflatex PerceivedIncomeRisk.tex

bibtex PerceivedIncomeRisk.aux
pdflatex PerceivedIncomeRisk.tex
pdflatex PerceivedIncomeRisk.tex

rm *.bbl *.aux *.blg *.log *.out *Notes.bib #*.tex

cd ..
cd ..
