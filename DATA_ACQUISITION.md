# Data acquisition and redistribution notes

## TrainTicket

The upstream benchmark is `https://github.com/FudanSELab/train-ticket` at commit
`e5ae0c3a4fa951980f9fe5640d6e077ab1852af9`. The package contains derived
prepared tensors and manifests for 150 controlled injections. The exact CPU,
memory, scratch-I/O, delay, and loss settings and split commands are recorded in
the manuscript and `manifests/cohort_protocol.json` in this archive. Preparation and deterministic
split source are under `scripts/care/`; reproducing extraction requires the
upstream raw telemetry described by those scripts.

## AIOps2022

The source is the public 2022 CCF International AIOps Challenge. Official portal
and materials links, selected cloud/date keys, filters, overlap grouping, and
source-file checksums are recorded in the cohort protocol and raw-source hash
manifest. The package contains a deterministic 128-incident prepared
reconstruction, not the unavailable earlier cohort.

## Third-party methods

Only adapters written for this study are included. Upstream sources with MIT or
Apache-2.0 licenses can be fetched from the URLs in the frozen protocol. Source
snapshots with no explicit license are not redistributed; their inspected
archive hashes are provided for provenance.
