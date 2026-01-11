from typing import Literal
import pandas
import seaborn
import matplotlib.pyplot as pyplot
from helpers.columns import ColumnName as Col

def plot_box_by_attrition(df: pandas.DataFrame,column_name: str,target: str = Col.ATTRITION,figsize=(6,4)):
    pyplot.figure(figsize=figsize)
    seaborn.boxplot(
        x=target,
        y=column_name,
        data=df
    )
    pyplot.title(f"{column_name} distribution by {target}")
    pyplot.xticks(rotation=0, ha="center")
    pyplot.show()


def plot(df: pandas.DataFrame, column_name: str, figsize=(4,4)):
    color = 'darkblue'
    pyplot.figure(figsize=figsize)
    seaborn.countplot(x=column_name, data=df)
    pyplot.title(column_name, color=color)
    pyplot.xlabel(column_name, color=color)
    pyplot.ylabel('frequency', color=color)
    if pandas.api.types.is_numeric_dtype(df[column_name]):
        pyplot.xticks(rotation=0, ha="center")
    else:
        pyplot.xticks(rotation=45, ha="right")
    
    pyplot.show()
    
def plot_hue(df: pandas.DataFrame, column_name: str, hue: str = Col.ATTRITION, figsize=(6,6)):
    color = 'darkblue'
    pyplot.figure(figsize=figsize)
    seaborn.countplot(x=column_name, hue=hue, data=df)
    pyplot.title(f"{column_name} by {hue}", color=color)
    pyplot.xlabel(column_name, color=color)
    pyplot.ylabel('frequency', color=color)
    if pandas.api.types.is_numeric_dtype(df[column_name]):
        pyplot.xticks(rotation=0, ha="center")
    else:
        pyplot.xticks(rotation=45, ha="right")
    pyplot.show()
    
def heatmap(df: pandas.DataFrame, figsize=(10,8)):
    color = 'darkblue'
    pyplot.figure(figsize=figsize)
    seaborn.heatmap(df.select_dtypes(include=["number","bool"]).corr(), annot=True, fmt=".2f", cmap='coolwarm', center=0)
    pyplot.title("Correlation Heatmap", color=color)
    pyplot.xticks(rotation=45, ha="right")
    pyplot.show()
def heatmap_target(df: pandas.DataFrame, target: str = Col.ATTRITION, figsize=(6,4)):
    color = 'darkblue'
    pyplot.figure(figsize=figsize)
    seaborn.heatmap(df.select_dtypes(include=["number","bool"]).corr()[[target]].sort_values(by=target, ascending=False), annot=True, fmt=".3f", cmap='coolwarm', center=0)
    pyplot.title(f"Correlation Heatmap with {target}", color=color)
    pyplot.xticks(rotation=45, ha="right")
    pyplot.show()
    
def plot_attrition_rate(df: pandas.DataFrame, column_name: str, figsize=(6,4), target: str = Col.ATTRITION):
    color = 'darkblue'
    tmp = (
        df
        .groupby(column_name)[target]
        .mean()
        .reset_index()
        .sort_values(target, ascending=False)
    )
    pyplot.figure(figsize=figsize)
    seaborn.barplot(x=column_name, y=target, data=tmp)
    pyplot.title(f"{target} rate by {column_name}", color=color)
    pyplot.xlabel(column_name, color=color)
    pyplot.ylabel(f"{target} rate", color=color)
    if pandas.api.types.is_numeric_dtype(df[column_name]):
        pyplot.xticks(rotation=0, ha="center")
    else:
        pyplot.xticks(rotation=45, ha="right")
    pyplot.show()
    

def plot_crosstab_heatmap(
    df: pandas.DataFrame,
    col_x: str,
    col_y: str = Col.ATTRITION,
    figsize= (6,4),
    normalize: Literal[0, 1, "index", "columns", "all"] | bool = "index"
):
    color = 'darkblue'
    ct = pandas.crosstab(
        df[col_x],
        df[col_y],
        normalize=normalize
    )
    pyplot.figure(figsize=figsize)
    seaborn.heatmap(ct, annot=True, fmt=".2f", cmap="coolwarm")
    pyplot.title(f"{col_x} vs {col_y}", color=color)
    pyplot.ylabel(col_x, color=color)
    pyplot.xlabel(col_y, color=color)
    if pandas.api.types.is_numeric_dtype(df[col_y]):
        pyplot.xticks(rotation=0, ha="center")
    else:
        pyplot.xticks(rotation=45, ha="right")
    if pandas.api.types.is_numeric_dtype(df[col_x]):
        pyplot.yticks(rotation=0, ha="center")
    else:
        pyplot.yticks(rotation=45, ha="right")
    pyplot.show()