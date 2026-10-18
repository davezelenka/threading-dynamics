# How Much Warmer? (v0.3.1) -- one-folder project

## One-time setup (Windows PowerShell)
Unzip this over (or next to) your existing Downloads\holocene_warmth_v03, then from that folder:

    cd C:\Users\DavidZelenka\Downloads\holocene_warmth_v03
    python code\project.py setup --old ..\holocene_proxy

setup creates data\, output\, climate\, figures\; moves the WorldClim files from code\climate to climate\
(deleting the duplicate prec subfolder); copies your GBIF downloads from ..\holocene_proxy\data;
puts the current data tables in data\. It never overwrites a data table you have edited.

## Run everything
    python code\project.py run

Order: fetch Pacific rhododendron -> clean-occ -> check-occ -> extract-climate (now with precipitation)
-> breadth -> site-bounds -> holocene-maps -> analogs -> profile (Port Angeles)
-> copy results into results\ and maps into figures\ -> make_tables.py -> make_figures.py.

    python code\project.py run --from site-bounds    # resume after a fix
    python code\project.py run --paper-only          # just rebuild tables\ and figures\ from results\
    python code\project.py run --no-fetch            # skip the GBIF download

## Folder layout
    main.tex, references.bib    paper           tables\   generated tables   figures\  generated charts
    results\   CSVs the paper reads (refreshed by run)
    code\      build_variability_proxy.py, make_tables.py, make_figures.py, project.py
    data\      species/site tables + occurrences\ + occurrences_clean\
    climate\   24 WorldClim GeoTIFFs              output\   everything the pipeline writes

## Overleaf
Upload main.tex, references.bib, tables\ and figures\. Placeholders fill automatically when a figure file exists.
