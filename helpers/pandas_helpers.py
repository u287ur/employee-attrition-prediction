import pandas

@pandas.api.extensions.register_dataframe_accessor("helpers")
class PandasExtensions:
    def __init__(self, pandas_obj)-> None: 
        self._obj = pandas_obj.copy()

    def get_single_value_columns(self)-> list:
        """Returns the columns that have only one unique value"""
        last_single_value_cols = [
            col for col in self._obj.columns if self._obj[col].nunique() == 1
        ]
        return last_single_value_cols

    def drop_single_value_columns(self)-> pandas.DataFrame:
        """Drops the columns that have only one unique value"""
        columns_to_drop = self.get_single_value_columns()
        if not columns_to_drop:
            return self._obj
        self._obj = self._obj.drop(columns = columns_to_drop)
        return self._obj

    def df(self):
        return self._obj
    
