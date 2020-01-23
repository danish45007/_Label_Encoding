from sklearn import preprocessing
import pandas as pd
"""
#- Label Encoding
- One Hot encoding
- Binarization


"""

class CategoricalFeatures:
    def __init__(self, df, categorical_features, encoding_type, handle_na=False):
        """
        df: pandas dataframe
        categorical_features: list of column names, e.g. ["ord_1", "nom_0"......]
        encoding_type: label, binary, ohe
        handle_na: True/False
        """
        self.df = df
        self.cat_feats = categorical_features
        self.enc_type = encoding_type
        self.handle_na = handle_na
        self.label_encoders = dict()
        self.binary_encoders = dict()
        self.ohe = None

        if self.handle_na:
            for c in self.cat_feats:
                self.df.loc[:, c] = self.df.loc[:, c].astype(str).fillna("-9999999")
        self.output_df = self.df.copy(deep=True)

    def _label_encoding(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(self.df[c].values)
            self.output_df.loc[:, c] = lbl.transform(self.df[c].values)
            self.label_encoders[c] = lbl
        return self.output_df

    def transform(self):
        if self.enc_type == "label":
            return self._label_encoding()
        else:
            raise Exception("Encoding Type not Understood")
if __name__ == "__main__":
    df = pd.read_csv("C:\\Users\\DANISH\\Downloads\\cat-in-the-dat-ii\\train.csv")
    cols = [c for c in df if c not in ["id","target"]]
    print(cols)
    cat_feats = CategoricalFeatures(df,categorical_features=cols,encoding_type="label",handle_na=True)
    out_ = cat_feats.transform()
    print(out_.head())
