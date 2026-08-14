# Bitcoin OTC trust network

- Source: Stanford Network Analysis Project (SNAP)
- Dataset page: https://snap.stanford.edu/data/soc-sign-bitcoinotc.html
- Download URL: https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz
- Local file: `soc-sign-bitcoinotc.csv.gz`
- SHA-256: `6424ac981dad3a019f697fc1b9fcd85c19d8d9f039797758e9ffb6fea100c373`

Each row has four comma-separated fields with no header:

1. source user ID
2. target user ID
3. trust rating, from -10 to 10
4. Unix timestamp in seconds

Local integrity audit:

- 35,592 rows
- 5,881 unique users
- 35,592 unique directed user pairs
- 32,029 positive and 3,563 negative ratings
- no malformed four-field rows
- no self-loops
- timestamps run from 2010-11-08 UTC to 2016-01-25 UTC

Because every ordered user pair occurs only once in this SNAP release, it supports temporal network-formation and cumulative social-capital analyses, but not within-dyad rating-revision analysis.
