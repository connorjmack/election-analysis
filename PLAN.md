# Implementation Plan: Multi-Factor Election Analysis Platform

**Generated:** 2026-01-17
**Request:** Build a precinct-level analysis system correlating election outcomes with education polarization, social media exposure/news deserts, and cost of living pressures
**Estimated Milestones:** 8
**Risk Level:** Medium - External data dependencies require API access or manual downloads; geographic joining at precinct level is complex

## Prerequisites
- [ ] Python 3.10+ installed
- [ ] Git submodules initialized (`git submodule update --init --recursive`)
- [ ] ~500MB disk space for processed data

## Architecture Decision Record
> **Approach:** Python-based ETL pipeline with pandas/geopandas, storing processed data in Parquet format. Analysis performed at county level (with precinct aggregation) due to external data availability constraints. Modular design allows adding data layers incrementally.

> **Key Trade-offs:**
> - County-level analysis (vs precinct): Most external datasets (ACS, BLS, news desert data) are only available at county level. We aggregate precinct votes to county, then join external factors.
> - Parquet over SQLite: Better for columnar analytics, pandas integration, and geographic data preservation.
> - Python over R: More ecosystem support for API access, web scraping, and ML pipelines.

> **Failure Domains:**
> - API rate limits on Census/BLS data → Mitigated by caching and incremental downloads
> - FIPS code mismatches between datasets → Mitigated by validation gates at each join
> - Memory constraints on full dataset → Mitigated by state-by-state processing

---

## Milestone 1: Initialize Project Structure

**Objective:** Create Python project skeleton with dependency management and directory structure

**Why This Order:** Foundation must exist before any data processing code

### Changes
- [ ] `pyproject.toml` - Create with dependencies: pandas, geopandas, pyarrow, requests, jupyter, matplotlib, seaborn, scikit-learn
- [ ] `src/__init__.py` - Create empty package marker
- [ ] `src/config.py` - Create with path constants (DATA_DIR, RAW_DIR, PROCESSED_DIR, EXTERNAL_DIR)
- [ ] `src/utils.py` - Create with FIPS code utilities (pad_fips, validate_county_fips)
- [ ] `notebooks/.gitkeep` - Create placeholder for analysis notebooks
- [ ] `data/external/.gitkeep` - Create placeholder for external datasets
- [ ] `data/processed/.gitkeep` - Create placeholder for processed outputs
- [ ] `.gitignore` - Add: `*.parquet`, `data/external/*.csv`, `data/processed/`, `__pycache__/`, `.venv/`, `*.ipynb_checkpoints/`

### Verification Gate
```bash
python -c "import sys; sys.path.insert(0, 'src'); from config import DATA_DIR; print(DATA_DIR)"
```

**Success Criteria:**
- [ ] Exit code: `0`
- [ ] Output contains path to `data` directory
- [ ] All directories exist: `src/`, `notebooks/`, `data/external/`, `data/processed/`

**Rollback if Failed:**
```bash
rm -rf src/ notebooks/ data/external/ data/processed/ pyproject.toml .gitignore
```

---

## Milestone 2: Build Election Data ETL Pipeline

**Objective:** Extract, transform, and aggregate precinct-level election data to county-level summaries

**Why This Order:** Election data is the foundation; all external data joins against it

### Changes
- [ ] `src/etl/__init__.py` - Create empty package marker
- [ ] `src/etl/election_loader.py` - Create module with functions:
  - `load_state_election(state_abbrev: str) -> pd.DataFrame` - Unzip and load single state CSV
  - `load_all_elections(year: int) -> pd.DataFrame` - Load all states for a year
  - `aggregate_to_county(df: pd.DataFrame) -> pd.DataFrame` - Sum votes by county_fips, party_simplified, office
- [ ] `src/etl/election_metrics.py` - Create module with functions:
  - `compute_vote_shares(df: pd.DataFrame) -> pd.DataFrame` - Calculate D/R/Other vote shares per county-office
  - `compute_swing(df_2018: pd.DataFrame, df_2022: pd.DataFrame) -> pd.DataFrame` - Calculate 2018→2022 vote swing
  - `compute_polarization_index(df: pd.DataFrame) -> pd.DataFrame` - Margin of victory as polarization proxy

### Verification Gate
```bash
python -c "
from src.etl.election_loader import load_state_election, aggregate_to_county
df = load_state_election('wy')  # Wyoming - smallest state
agg = aggregate_to_county(df)
assert 'county_fips' in agg.columns
assert len(agg) > 0
print(f'SUCCESS: {len(agg)} county-office rows from Wyoming')
"
```

**Success Criteria:**
- [ ] Exit code: `0`
- [ ] Output contains: `SUCCESS:`
- [ ] Wyoming data loads without error

**Rollback if Failed:**
```bash
rm -rf src/etl/
```

---

## Milestone 3: Integrate Education Data (ACS)

**Objective:** Download and process American Community Survey education attainment data at county level

**Why This Order:** Education polarization is first analysis dimension; establishes pattern for other external data

### Changes
- [ ] `src/external/__init__.py` - Create empty package marker
- [ ] `src/external/census_api.py` - Create module with functions:
  - `get_acs_education(year: int) -> pd.DataFrame` - Fetch ACS 5-year estimates for educational attainment (B15003 table)
  - `compute_college_rate(df: pd.DataFrame) -> pd.DataFrame` - Calculate % with bachelor's degree or higher per county
- [ ] `src/external/data_sources.md` - Document data source URLs, API endpoints, and field mappings
- [ ] `data/external/acs_education_2022.parquet` - Cached education data (generated by pipeline)

### Verification Gate
```bash
python -c "
from src.external.census_api import get_acs_education, compute_college_rate
# Test with cached/mock data if API unavailable
import os
if os.path.exists('data/external/acs_education_2022.parquet'):
    import pandas as pd
    df = pd.read_parquet('data/external/acs_education_2022.parquet')
    assert 'county_fips' in df.columns
    assert 'college_rate' in df.columns
    print(f'SUCCESS: {len(df)} counties with education data')
else:
    print('SKIP: Run full pipeline to generate cached data')
"
```

**Success Criteria:**
- [ ] Exit code: `0`
- [ ] Education data contains `county_fips` and `college_rate` columns
- [ ] Coverage: 3000+ counties (US has ~3143)

**Rollback if Failed:**
```bash
rm -rf src/external/ data/external/acs_*.parquet
```

---

## Milestone 4: Integrate News Desert Data

**Objective:** Incorporate UNC Hussman School news desert dataset mapping local news availability

**Why This Order:** News desert data is static/downloadable, lower complexity than economic APIs

### Changes
- [ ] `src/external/news_deserts.py` - Create module with functions:
  - `load_news_desert_data() -> pd.DataFrame` - Load UNC news desert dataset (manual download required)
  - `compute_news_access_score(df: pd.DataFrame) -> pd.DataFrame` - Create composite score: newspapers per capita, digital-only outlets, news desert flag
- [ ] `src/external/data_sources.md` - Append news desert data source documentation
- [ ] `data/external/README.md` - Create with manual download instructions for UNC data (https://www.usnewsdeserts.com/download-data/)

### Verification Gate
```bash
python -c "
import os
if os.path.exists('data/external/news_deserts.csv') or os.path.exists('data/external/news_deserts.parquet'):
    from src.external.news_deserts import load_news_desert_data
    df = load_news_desert_data()
    assert 'county_fips' in df.columns
    print(f'SUCCESS: {len(df)} counties with news data')
else:
    print('PENDING: Download news desert data per data/external/README.md')
"
```

**Success Criteria:**
- [ ] Exit code: `0`
- [ ] News desert data loadable with county_fips key
- [ ] Includes binary news_desert flag and newspaper count

**Rollback if Failed:**
```bash
rm src/external/news_deserts.py
```

---

## Milestone 5: Integrate Cost of Living Data

**Objective:** Add BLS regional price parities and housing cost indices at county/metro level

**Why This Order:** Completes the three analysis dimensions before building unified dataset

### Changes
- [ ] `src/external/bls_data.py` - Create module with functions:
  - `get_regional_price_parity() -> pd.DataFrame` - Load BEA Regional Price Parities (RPP) by metro/non-metro
  - `get_housing_costs() -> pd.DataFrame` - Load HUD Fair Market Rents or Zillow ZHVI by county
  - `compute_affordability_index(df: pd.DataFrame, income_df: pd.DataFrame) -> pd.DataFrame` - Housing cost / median income ratio
- [ ] `src/external/data_sources.md` - Append BLS/BEA/HUD data source documentation

### Verification Gate
```bash
python -c "
from src.external.bls_data import get_housing_costs
import os
if os.path.exists('data/external/housing_costs.parquet'):
    import pandas as pd
    df = pd.read_parquet('data/external/housing_costs.parquet')
    assert 'county_fips' in df.columns or 'metro_code' in df.columns
    print(f'SUCCESS: {len(df)} geographic units with cost data')
else:
    print('PENDING: Run pipeline to fetch cost of living data')
"
```

**Success Criteria:**
- [ ] Exit code: `0`
- [ ] Cost data contains geographic identifier joinable to election data
- [ ] Includes housing cost metric (rent or home value)

**Rollback if Failed:**
```bash
rm src/external/bls_data.py data/external/housing_*.parquet
```

---

## Milestone 6: Build Unified Analysis Dataset

**Objective:** Join election results with all external factors into single analysis-ready dataset

**Why This Order:** All component datasets must exist before joining

### Changes
- [ ] `src/analysis/__init__.py` - Create empty package marker
- [ ] `src/analysis/build_dataset.py` - Create module with functions:
  - `build_unified_dataset(year: int) -> pd.DataFrame` - Join election + education + news + cost data on county_fips
  - `validate_coverage(df: pd.DataFrame) -> dict` - Report missing data by column/state
  - `export_analysis_dataset(df: pd.DataFrame, path: str)` - Save to Parquet with metadata
- [ ] `data/processed/unified_2022.parquet` - Generated unified dataset

### Verification Gate
```bash
python -c "
import pandas as pd
import os
if os.path.exists('data/processed/unified_2022.parquet'):
    df = pd.read_parquet('data/processed/unified_2022.parquet')
    required = ['county_fips', 'dem_share', 'college_rate']
    missing = [c for c in required if c not in df.columns]
    assert not missing, f'Missing columns: {missing}'
    print(f'SUCCESS: Unified dataset with {len(df)} rows, {len(df.columns)} columns')
else:
    print('PENDING: Run build_dataset.py to generate unified data')
"
```

**Success Criteria:**
- [ ] Exit code: `0`
- [ ] Unified dataset contains election + at least one external factor
- [ ] County FIPS codes are consistent (5-digit, zero-padded strings)
- [ ] No duplicate county rows (one row per county per office)

**Rollback if Failed:**
```bash
rm -rf src/analysis/ data/processed/unified_*.parquet
```

---

## Milestone 7: Create Analysis Notebooks

**Objective:** Build Jupyter notebooks for exploratory analysis and visualization

**Why This Order:** Unified dataset must exist before analysis

### Changes
- [ ] `notebooks/01_data_exploration.ipynb` - Create notebook:
  - Load unified dataset
  - Summary statistics by state
  - Missing data heatmap
  - Distribution plots for each variable
- [ ] `notebooks/02_correlation_analysis.ipynb` - Create notebook:
  - Correlation matrix: vote share vs education, news access, cost of living
  - Scatter plots with regression lines
  - State-level vs national patterns
- [ ] `notebooks/03_geographic_visualization.ipynb` - Create notebook:
  - Choropleth maps of each variable
  - Bivariate maps (e.g., education + vote share)
  - Requires geopandas and county shapefiles

### Verification Gate
```bash
python -c "
import os
notebooks = ['notebooks/01_data_exploration.ipynb', 'notebooks/02_correlation_analysis.ipynb']
existing = [n for n in notebooks if os.path.exists(n)]
print(f'SUCCESS: {len(existing)}/{len(notebooks)} notebooks created')
"
```

**Success Criteria:**
- [ ] Exit code: `0`
- [ ] At least 2 notebooks exist
- [ ] Notebooks are valid JSON (parseable as Jupyter format)

**Rollback if Failed:**
```bash
rm -f notebooks/*.ipynb
```

---

## Milestone 8: Build Regression Analysis Pipeline

**Objective:** Create statistical models quantifying factor impacts on vote share/swing

**Why This Order:** Final milestone; requires all preceding data and exploration

### Changes
- [ ] `src/analysis/models.py` - Create module with functions:
  - `fit_ols_model(df: pd.DataFrame, outcome: str, predictors: list) -> statsmodels.Result` - OLS regression with robust standard errors
  - `fit_spatial_lag_model(df: pd.DataFrame, ...) -> Result` - Spatial regression accounting for neighbor effects
  - `compute_variable_importance(model) -> pd.DataFrame` - Standardized coefficients and confidence intervals
- [ ] `src/analysis/reporting.py` - Create module with functions:
  - `generate_regression_table(models: list) -> str` - LaTeX/markdown regression table
  - `plot_coefficients(model) -> matplotlib.Figure` - Coefficient plot with CIs
- [ ] `notebooks/04_regression_analysis.ipynb` - Create notebook:
  - Full regression analysis with controls
  - Interaction effects (e.g., education × news desert)
  - Robustness checks

### Verification Gate
```bash
python -c "
from src.analysis.models import fit_ols_model
import pandas as pd
# Quick test with synthetic data
import numpy as np
np.random.seed(42)
test_df = pd.DataFrame({
    'dem_share': np.random.uniform(0.3, 0.7, 100),
    'college_rate': np.random.uniform(0.1, 0.5, 100),
    'news_score': np.random.uniform(0, 1, 100)
})
result = fit_ols_model(test_df, 'dem_share', ['college_rate', 'news_score'])
assert result is not None
print('SUCCESS: OLS model fitting works')
"
```

**Success Criteria:**
- [ ] Exit code: `0`
- [ ] Model fitting produces valid results object
- [ ] Coefficient extraction works

**Rollback if Failed:**
```bash
rm src/analysis/models.py src/analysis/reporting.py notebooks/04_regression_analysis.ipynb
```

---

## Final Verification

```bash
# Full integration check:
python -c "
import os
import sys

checks = [
    ('Project structure', os.path.exists('src/config.py')),
    ('ETL pipeline', os.path.exists('src/etl/election_loader.py')),
    ('External data modules', os.path.exists('src/external/census_api.py')),
    ('Analysis pipeline', os.path.exists('src/analysis/build_dataset.py')),
    ('Notebooks exist', os.path.exists('notebooks/01_data_exploration.ipynb')),
]

passed = sum(1 for _, v in checks if v)
print(f'Integration check: {passed}/{len(checks)} components present')

for name, status in checks:
    print(f'  [{\"✓\" if status else \"✗\"}] {name}')

sys.exit(0 if passed == len(checks) else 1)
"
```

**Definition of Done:**
- [ ] All milestone gates passed
- [ ] Election data loads and aggregates to county level
- [ ] At least one external data source (education) successfully joined
- [ ] Correlation analysis notebook produces visualizations
- [ ] Regression model quantifies relationship between factors and vote share

## Appendix: File Change Summary

| File | Action | Milestone |
|------|--------|-----------|
| `pyproject.toml` | Create | 1 |
| `src/__init__.py` | Create | 1 |
| `src/config.py` | Create | 1 |
| `src/utils.py` | Create | 1 |
| `.gitignore` | Create | 1 |
| `src/etl/__init__.py` | Create | 2 |
| `src/etl/election_loader.py` | Create | 2 |
| `src/etl/election_metrics.py` | Create | 2 |
| `src/external/__init__.py` | Create | 3 |
| `src/external/census_api.py` | Create | 3 |
| `src/external/data_sources.md` | Create | 3 |
| `src/external/news_deserts.py` | Create | 4 |
| `data/external/README.md` | Create | 4 |
| `src/external/bls_data.py` | Create | 5 |
| `src/analysis/__init__.py` | Create | 6 |
| `src/analysis/build_dataset.py` | Create | 6 |
| `notebooks/01_data_exploration.ipynb` | Create | 7 |
| `notebooks/02_correlation_analysis.ipynb` | Create | 7 |
| `notebooks/03_geographic_visualization.ipynb` | Create | 7 |
| `src/analysis/models.py` | Create | 8 |
| `src/analysis/reporting.py` | Create | 8 |
| `notebooks/04_regression_analysis.ipynb` | Create | 8 |

## Appendix: External Data Sources

| Dataset | Source | Geographic Level | URL |
|---------|--------|------------------|-----|
| Educational Attainment | ACS 5-Year (B15003) | County | Census API |
| News Deserts | UNC Hussman School | County | usnewsdeserts.com |
| Regional Price Parity | BEA | Metro/Non-metro | bea.gov |
| Fair Market Rents | HUD | County | huduser.gov |
| County Shapefiles | Census TIGER | County | census.gov |
