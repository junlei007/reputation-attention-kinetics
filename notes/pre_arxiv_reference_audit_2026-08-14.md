# Pre-arXiv reference and wording audit (2026-08-14)

## Scope and method

- Checked every citation key in `paper/main.tex` against its manual
  bibliography entry; all 21 cited keys have exactly one entry and there are
  no uncited entries.
- Checked the 19 journal/conference DOIs against Crossref registration data
  and publisher records. The Journal of Web Science DOI is not exposed as a
  Crossref JSON record, so it was verified directly on the journal's article
  page and PDF.
- Checked the SNAP dataset citation against Stanford's official dataset pages
  and the software citation against the versioned Zenodo record.
- Read each citation context in the Introduction and Discussion to assess
  whether the cited source supports the adjacent claim.
- Searched the manuscript, supplement and proof appendix for placeholders,
  undisclosed tool names, stock AI phrasing, unsupported priority claims and
  over-strong causal or validation language.

## Result

No fabricated or non-resolving reference was found. Titles, authors,
publication venues, volumes, issues, page/article numbers and publication
years agree with the registered or publisher metadata. Online-first years for
several Cambridge/OUP articles precede their issue years; the manuscript
correctly uses the year of the cited journal volume.

Verified citation keys (21/21): `alaa2018`, `gargiulo2019`, `ubaldi2016`,
`raynal2026`, `albi2025`, `burger2022`, `nurisso2026`, `bcw2023`,
`during2024`, `allmeier2025`, `arena2023`, `arena2026`, `dfh2016`,
`ditlevsen2017`, `fournier2015`, `lerner2025`, `kumar2016`, `butts2008`,
`snijders2017`, `du2026software`, and `snap`.

One citation-use issue was corrected: Snijders (2017) concerns stochastic
actor-oriented models for panel network data, not a relational-event model.
The Introduction now presents it as a complementary framework rather than
grouping it with Butts (2008) and the Arena et al. REM papers. Related stock or
over-strong wording (`novel`, `unambiguous`, `honest`, and the shorthand
`significant, right sign`) was also replaced with claim-specific language.

The repository-level `references.bib` had lagged behind the manual
bibliography in `paper/main.tex`. It has been synchronized with all cited
works and supplemented with missing issue metadata where the publisher record
provides it.

## Source-level checks of potentially sensitive claims

- Stanford SNAP confirms 5,881 nodes, 35,592 directed weighted edges and the
  -10 to +10 rating range for Bitcoin OTC; it identifies Kumar et al. (2016)
  as a dataset citation.
- Gargiulo, Bertazzi and Huet (2019) explicitly report increasing reputation
  inequality toward a high steady value in Bitcoin OTC.
- Arena, Mulder and Leenders (2023) estimate parametric memory-decay functions
  in relational-event models; their 2026 paper separately models positive and
  negative event types and estimates type-specific memory parameters.
- Allmeier and Gast (2025) assume a finite state space in their main graphon
  mean-field result and report deterministic and random-graph approximation
  orders used in the proof appendix's comparison.
- Fournier and Guillin (2015), Theorem 1, gives the logarithmic correction at
  the critical case p=d/2, supporting the conservative d=2, p=1 bound used in
  the manuscript.
- Lerner, Hancean and Perc (2025) use relational hyperevent models to define
  tailored null distributions for time-stamped hyperedges, supporting the
  Introduction's description.

## arXiv/AI note

arXiv's [current moderation guidance](https://info.arxiv.org/help/moderation/)
explicitly says that significant use of
text-to-text generative AI should be reported consistently with subject
standards and that every named author remains responsible for errors,
plagiarized material, incorrect references or misleading content regardless
of how it was generated. The guidance does not state that disclosed use is
itself grounds for rejection, and no official policy was located describing a
general-purpose AI-text detector. The manuscript's explicit AI-use disclosure
and this source-by-source verification address the stated policy risks without
making an unsupported claim about detection.
