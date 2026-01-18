# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a data analysis repository for US election results. It uses git submodules to include official precinct-level election data from MIT Election Data and Science Lab (MEDSL):

- `data/2018-elections-official/` - 2018 precinct data (deprecated, now on Harvard Dataverse)
- `data/2022-elections-official/` - 2022 precinct data with complete coverage of all 50 states + DC

## Data Structure

Election data files are stored as zipped CSVs in `individual_states/` directories with naming pattern: `{year}-{state}-local-precinct-general.zip`

### Key Data Fields

All election data follows a standardized schema (see `data/2022-elections-official/codebook.md`):
- `precinct` - Smallest election reporting unit; "FLOATING" suffix indicates aggregated data
- `office` - Standardized: US PRESIDENT, US SENATE, US HOUSE, GOVERNOR, STATE SENATE, STATE HOUSE
- `party_detailed` / `party_simplified` - DEMOCRAT, REPUBLICAN, LIBERTARIAN, OTHER, NONPARTISAN
- `mode` - Voting mode (TOTAL, ABSENTEE, PROVISIONAL, etc.)
- `votes` - Numeric vote count
- `county_fips` / `jurisdiction_fips` - Census FIPS codes (string-padded)
- `district` - 3-digit padded for legislative/congressional races, "statewide" for statewide offices
- `dataverse` - PRESIDENT, SENATE, HOUSE, or STATE (for state-level offices)

## Working with Submodules

```bash
# Initialize and fetch submodule data
git submodule update --init --recursive
```

## R Scripts

The `repository_code/` directory contains R scripts for generating visualizations:
- `hexmap.r` - Generates US hexmap showing data collection progress (requires sp, tidyverse, broom, rgeos, rgdal)

## Data Quality Notes

- Some states have counties missing precinct-level data (notably Indiana, Arkansas)
- Zero-vote rows may be fictitious (candidates not on ballot in that precinct)
- Mode reporting varies by state - check `mode` field carefully to avoid double-counting with TOTAL rows
- State-specific caveats are documented in the 2022 README.md
