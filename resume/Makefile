TECTONIC ?= tectonic
SOURCE := mmayer.tex
OUTPUT_DIR := build
OUTPUT_PDF := $(OUTPUT_DIR)/mmayer.pdf
SITE_PDF := ../static/files/resume.pdf
SOURCES := $(SOURCE) altacv.cls page1sidebar.tex page2sidebar.tex

.PHONY: build publish clean

build: $(OUTPUT_PDF)

$(OUTPUT_PDF): $(SOURCES)
	mkdir -p $(OUTPUT_DIR)
	$(TECTONIC) --outdir $(OUTPUT_DIR) $(SOURCE)

publish: build
	cp $(OUTPUT_PDF) $(SITE_PDF)

clean:
	$(RM) -r $(OUTPUT_DIR)
