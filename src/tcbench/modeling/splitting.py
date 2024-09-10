from __future__ import annotations

import polars as pl

def _verify_split_indices_table(
    df: pl.DataFrame, 
    y_colname:str, 
    expected_test_size: float = 0.1
) -> bool:
    df_split_sizes = (df
        .group_by(y_colname, "split")
        .agg(
            pl.col("row_id").count()
        )
        .pivot(
            index=y_colname,
            on="split",
            values="row_id"
        )
        .with_columns(
            total=(
                pl.sum_horizontal("test", "train")
            )
        )
        .with_columns(
            test_size=(
                pl.col("test") / pl.col("total")
            ),
            train_size=(
                pl.col("train") / pl.col("total")
            )
        )
        .sort("total", descending=True)
    )
    
    df_tmp = df_split_sizes.filter(pl.col("test_size") < expected_test_size)
    assert df_tmp.is_empty(), \
        "some classes have less test sample than expected" + "\n" + str(df_tmp)
    
    df_tmp = df_split_sizes.filter(pl.col("train_size") == 0)
    assert df_tmp.is_empty(), \
        "some classes have no train samples" + "\n" + str(df_tmp)

    return True

def _get_split_indices(
    df: pl.DataFrame, 
    y_colname: str = "app",
    index_colname: str = "row_id", 
    test_size: float = 0.1,
    seed: int = 1, 
) -> pl.DataFrame:

    df_split_indices = (df
        .select(
            y_colname,
            index_colname,
        )
        .group_by(y_colname)
        .map_groups(
            lambda df_group: (df_group
                # new index, specific for each label
                .with_row_index("_inner_idx")
                # shuffle original index
                .with_columns(
                    pl.col(index_colname).shuffle(seed=seed)    
                )
                .with_columns(
                    # split=test for the first test_size samples
                    # split=train otherwise
                    split=(
                        pl.when(
                            pl.col("_inner_idx") 
                            < pl.lit(len(df_group) * test_size).ceil()
                        )
                        .then(pl.lit("test"))
                        .otherwise(pl.lit("train"))
                    )
                )
                .drop("_inner_idx")
            )
        )
    )    

    _verify_split_indices_table(
        df_split_indices, 
        y_colname,
        expected_test_size=test_size
    )

    return df_split_indices

def add_split_column(
    df: pl.DataFrame, 
    y_colname: str = "app",
    index_colname: str = "row_id", 
    test_size: float = 0.1,
    seed: int = 1, 
) -> pl.DataFrame:
    df_split_indices = _get_split_indices(df, y_colname, index_colname, test_size, seed)
    return df.join(
        df_split_indices, 
        on=[y_colname, index_colname]
    )

