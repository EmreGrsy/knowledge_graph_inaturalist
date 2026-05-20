.PHONY: schema load validate run clean help

SCHEMA := kg/schema.yaml
OWL    := kg/schema.owl
SHAPES := kg/shapes.ttl
DATA   := data/observations.ttl

help:
	@echo "Targets:"
	@echo "  make schema    Generate OWL + SHACL from LinkML"
	@echo "  make load      Fetch iNaturalist data, validate, serialize to Turtle"
	@echo "  make validate  Re-run SHACL validation on the existing data file"
	@echo "  make run       Start the Streamlit app"
	@echo "  make clean     Remove generated artifacts and logs"

$(OWL): $(SCHEMA)
	gen-owl $(SCHEMA) > $(OWL)

$(SHAPES): $(SCHEMA)
	gen-shacl $(SCHEMA) > $(SHAPES)

schema: $(OWL) $(SHAPES)

load: $(SHAPES)
	python -m kg.load

validate: $(SHAPES) $(DATA)
	pyshacl -s $(SHAPES) -d $(DATA)

run:
	streamlit run app.py

clean:
	rm -f $(OWL) $(SHAPES)
	rm -f logs/*.log logs/*.log.*
