#!/bin/bash

cd WorkingFolder/PythonCode/


ipython PerceivedIncomeRisk.py
jupyter nbconvert --output-dir='./latex/' --to=latex --LatexExporter.template_file=./latex/PerceivedIncomeRisk0.tplx TagRemovePreprocessor.remove_cell_tags='{"remove_cell"}' PerceivedIncomeRiskAttribution.ipynb

ipython TexTablesMover_att.py

cd ./latex/

pdflatex PerceivedIncomeRiskAttribution.tex

bibtex PerceivedIncomeRiskAttribution.aux
pdflatex PerceivedIncomeRiskAttribution.tex
pdflatex PerceivedIncomeRiskAttribution.tex

rm *.bbl *.aux *.blg *.log *.out *Notes.bib #*.tex

cd ..
cd ..
