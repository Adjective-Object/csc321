all: writeup.pdf

%.pdf: %.md
	pandoc -S \
		-o $@ \
		-V geometry:margin=1in\
		--template=paper-template.latex \
		$<
%.tex: %.md
	pandoc -S \
		-o $@ \
		-V geometry:margin=1in\
		--template=paper-template.latex \
		$<

watch: watch-all

watch-%:
	while true; do \
		make --no-print-directory $*; \
		fswatch --event Updated -x -1 ./$*.md ./$*.bib; \
	done

clean:
	rm *.pdf
