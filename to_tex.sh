#!/bin/bash

cd WorkingFolder/PythonCode/

jupyter nbconvert PerceivedIncomeRisk.ipynb --to latex
sed -r -i 's/documentclass\[15pt\]\{article\}/documentclass[8pt]{extarticle}/' PerceivedIncomeRisk.tex
sed -r -i 's/geometry\{verbose,tmargin=1in,bmargin=1in,lmargin=1in,rmargin=1in}/geometry{verbose,tmargin=1in,bmargin=1in,lmargin=0.2in,rmargin=0.2in}/' PerceivedIncomeRisk.tex

pdflatex PerceivedIncomeRisk.tex