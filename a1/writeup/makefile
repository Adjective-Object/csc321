all: writeup.pdf

%.pdf: %.md %-library.bib
	pandoc -S \
		-o $@ \
		-V geometry:margin=1in\
		--filter pandoc-citeproc \
		--template=paper-template.latex \
		$<

watch:
	while true; do \
		make --no-print-directory all; \
		fswatch --event Updated -x -1 ./*.md ./*.bib; \
	done
