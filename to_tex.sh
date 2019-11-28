#!/bin/bash

cd WorkingFolder/PythonCode/


ipython PerceivedIncomeRisk.py
jupyter nbconvert --to=latex --LatexExporter.template_file=./PerceivedIncomeRisk.tplx PerceivedIncomeRisk.ipynb

pdflatex PerceivedIncomeRisk.tex

bibtex PerceivedIncomeRisk.aux
pdflatex PerceivedIncomeRisk.tex
pdflatex PerceivedIncomeRisk.tex

rm *.bbl *.aux *.blg *.log *.out *Notes.bib #*.tex

cd ..
cd .. 