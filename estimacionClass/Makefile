DOCS = Actividad1_PlantamientoProblema \
       Actividad2_Modelado \
       Actividad3_MetodologiaSolucion \
       Actividad4_Implementacion \
       PIA_ResumenCasoEstudio

PDFS = $(addsuffix .pdf, $(DOCS))

LATEXMK = latexmk -pdf -interaction=nonstopmode -halt-on-error

.PHONY: all clean

all: $(PDFS)

%.pdf: %.tex
	$(LATEXMK) $<

clean:
	latexmk -C
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.fls *.fdb_latexmk
