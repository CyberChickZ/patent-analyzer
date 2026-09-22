"""One-off loader: Reliance on Science pcs_oa_uspto.csv (Zenodo 21493744)
-> GCS -> amie_patents.pcs_oa_raw -> amie_patents.pcs_oa (bucketed on
oa_id) + amie_patents.pcs_oa_by_patent (bucketed on patent_pub).

    PYTHONPATH=. python scripts/load_pcs_oa.py /tmp/ros/pcs_oa_uspto.csv [--skip-upload] [--skip-load]

Raw columns (header verified 2026-09-18): reftype,confscore,oaid,patent,wherefound
  oaid    digits only ('3066') -> oa_id 'W3066' (relos DOCUMENTATION.md says
          newer exports are 'W…'-prefixed, so both spellings are accepted)
  patent  'us-10494607-b2' (country-number-kind, lower case; reissues
          'us-re39456-e', designs 'us-d…-s') -> UPPER = pubs.publication_number
  grant_year / family_id come from a LEFT JOIN on amie_patents.pubs
  (1995+ only; older grants get NULL).
"""
import os
import sys
import time

from google.cloud import bigquery, storage

PROJECT = os.getenv("GC_PROJECT", "aime-hello-world")
BUCKET = os.getenv("GCS_BUCKET", "aime-hello-world-amie-uswest1")
OBJECT = "ros/pcs_oa_uspto.csv"
DS = f"{PROJECT}.amie_patents"

RAW_SCHEMA = [bigquery.SchemaField(n, t) for n, t in (
    ("reftype", "STRING"), ("confscore", "INT64"), ("oaid", "STRING"), ("patent", "STRING"), ("wherefound", "STRING"))]

CTAS_OA = f"""
CREATE OR REPLACE TABLE `{DS}.pcs_oa`
PARTITION BY RANGE_BUCKET(bucket, GENERATE_ARRAY(0, 4000, 1))
CLUSTER BY oa_id AS
WITH r AS (
  SELECT IF(STARTS_WITH(UPPER(oaid), 'W'), UPPER(oaid), CONCAT('W', oaid)) AS oa_id,
         UPPER(TRIM(patent)) AS patent_pub, LOWER(reftype) AS reftype, confscore, LOWER(wherefound) AS wherefound
  FROM `{DS}.pcs_oa_raw`)
SELECT MOD(ABS(FARM_FINGERPRINT(r.oa_id)), 4000) AS bucket, r.oa_id, r.patent_pub, r.reftype, r.confscore, r.wherefound,
       DIV(p.publication_date, 10000) AS grant_year, p.family_id
FROM r LEFT JOIN (SELECT publication_number, publication_date, family_id FROM `{DS}.pubs` WHERE country_code = 'US') p
  ON p.publication_number = r.patent_pub
"""

CTAS_BY_PATENT = f"""
CREATE OR REPLACE TABLE `{DS}.pcs_oa_by_patent`
PARTITION BY RANGE_BUCKET(bucket, GENERATE_ARRAY(0, 4000, 1))
CLUSTER BY patent_pub AS
SELECT MOD(ABS(FARM_FINGERPRINT(patent_pub)), 4000) AS bucket, patent_pub, oa_id, reftype, confscore, wherefound,
       grant_year, family_id
FROM `{DS}.pcs_oa`
"""


def run_sql(bq, label, sql):
    dry = bq.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False))
    est = dry.total_bytes_processed / 2 ** 30
    print(f"[{label}] dry-run estimate {est:.2f} GiB", flush=True)
    t = time.time()
    job = bq.query(sql, job_config=bigquery.QueryJobConfig(maximum_bytes_billed=int((est + 2) * 2 ** 30)))
    job.result()
    print(f"[{label}] done in {time.time() - t:.0f}s, processed {job.total_bytes_processed / 2 ** 30:.2f} GiB, "
          f"billed {job.total_bytes_billed / 2 ** 30:.2f} GiB", flush=True)


def main(path, skip_upload=False, skip_load=False):
    bq = bigquery.Client(project=PROJECT)
    uri = f"gs://{BUCKET}/{OBJECT}"
    if not skip_upload:
        blob = storage.Client(project=PROJECT).bucket(BUCKET).blob(OBJECT)
        blob.chunk_size = 64 * 2 ** 20
        t = time.time()
        blob.upload_from_filename(path, timeout=600)
        print(f"[upload] {os.path.getsize(path) / 2 ** 30:.2f} GiB -> {uri} in {time.time() - t:.0f}s", flush=True)
    if not skip_load:
        cfg = bigquery.LoadJobConfig(source_format=bigquery.SourceFormat.CSV, skip_leading_rows=1, schema=RAW_SCHEMA,
                                     write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE)
        t = time.time()
        job = bq.load_table_from_uri(uri, f"{DS}.pcs_oa_raw", job_config=cfg)
        job.result()
        tb = bq.get_table(f"{DS}.pcs_oa_raw")
        print(f"[load] pcs_oa_raw {tb.num_rows} rows, {tb.num_bytes / 2 ** 30:.2f} GiB in {time.time() - t:.0f}s "
              f"(load jobs are free)", flush=True)
    run_sql(bq, "ctas pcs_oa", CTAS_OA)
    run_sql(bq, "ctas pcs_oa_by_patent", CTAS_BY_PATENT)
    for name in ("pcs_oa", "pcs_oa_by_patent"):
        tb = bq.get_table(f"{DS}.{name}")
        print(f"[table] {name}: {tb.num_rows} rows, {tb.num_bytes / 2 ** 30:.2f} GiB, partition={tb.range_partitioning.field}, "
              f"cluster={tb.clustering_fields}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0] if args else "/tmp/ros/pcs_oa_uspto.csv",
         skip_upload="--skip-upload" in sys.argv, skip_load="--skip-load" in sys.argv)
