## 2026-03-03 — Project Setup

### Objective
Set up dependency and ingestion layer.

### Done
- Install Poetry
- Configure pyproject.toml (PEP 621 format)
- Create SQLite ingestion script
- Validate DB creation

### Outcome
- churn.db created successfully
- customers_raw table: 7043 rows

### Notes
- authors field must be array
- Use ^3.12 instead of exact patch

### Next
- Implement split module.