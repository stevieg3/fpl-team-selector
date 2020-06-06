import os

from s3fs import S3FileSystem
import pyarrow as pa
import pyarrow.parquet as pq

AWS_ACCESS_KEY_ID = os.environ['AWS_ACCESS_KEY_ID']

AWS_SECRET_KEY = os.environ['AWS_SECRET_KEY']

S3_BUCKET_PATH = "s3://fpl-analysis-data"

GW_PREDICTIONS_SUFFIX = "/gw_predictions"

GW_RETRO_PREDICTIONS_SUFFIX = "/gw_retro_predictions"

s3_filesystem = S3FileSystem(
    key=AWS_ACCESS_KEY_ID,
    secret=AWS_SECRET_KEY
)


def write_dataframe_to_s3(df, s3_root_path, partition_cols, filesystem=s3_filesystem):
    """
    Write Pandas DataFrame as parquet file in S3.

    :param df: Pandas DataFrame
    :param s3_root_path: S3 root path
    :param partition_cols: Columns to partition parquet file by
    :param filesystem: S3FileSystem object
    :return: None
    """
    arrow_table = pa.Table.from_pandas(
        df,
        preserve_index=False
    )

    pq.write_to_dataset(
        arrow_table,
        root_path=s3_root_path,
        partition_cols=partition_cols,
        filesystem=filesystem
    )
