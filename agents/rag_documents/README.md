# RAG governance corpus

Place the two public PDF documents below in this folder before running `9_ingest_pinecone.py`.

Expected filenames:

- `tec_student_privacy_notice.pdf`
  Official Tecnológico de Monterrey / ITESM student privacy notice:
  https://egade.tec.mx/sites/default/files/static/Aviso-de-Privacidad-Alumnos.pdf

- `nist_privacy_framework_v1.pdf`
  NIST Privacy Framework v1.0:
  https://doi.org/10.6028/NIST.CSWP.01162020

The other files in this folder are synthetic teaching material:
- `student_data_catalog.csv`
- `ai_use_policy.txt`
- `retention_policy.txt`

They are intentionally mixed with public policy documents so students can see RAG
over heterogeneous sources: external framework, privacy notice, internal policy,
and structured metadata.
