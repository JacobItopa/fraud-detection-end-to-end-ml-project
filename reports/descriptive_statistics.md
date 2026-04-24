# Exploratory Data Analysis: Descriptive Statistics

## 1. Dataset Shape
- **Rows:** 6,362,620
- **Columns:** 11

## 2. Column Data Types
```text
<class 'pandas.DataFrame'>
RangeIndex: 6362620 entries, 0 to 6362619
Data columns (total 11 columns):
 #   Column          Dtype  
---  ------          -----  
 0   step            int64  
 1   type            str    
 2   amount          float64
 3   nameOrig        str    
 4   oldbalanceOrg   float64
 5   newbalanceOrig  float64
 6   nameDest        str    
 7   oldbalanceDest  float64
 8   newbalanceDest  float64
 9   isFraud         int64  
 10  isFlaggedFraud  int64  
dtypes: float64(5), int64(3), str(3)
memory usage: 534.0 MB
```

## 3. Missing Values
|                |   Missing Values |
|:---------------|-----------------:|
| step           |                0 |
| type           |                0 |
| amount         |                0 |
| nameOrig       |                0 |
| oldbalanceOrg  |                0 |
| newbalanceOrig |                0 |
| nameDest       |                0 |
| oldbalanceDest |                0 |
| newbalanceDest |                0 |
| isFraud        |                0 |
| isFlaggedFraud |                0 |

## 4. Summary Statistics (Numerical)
|                |       count |             mean |              std |   min |     25% |      50% |              75% |           max |
|:---------------|------------:|-----------------:|-----------------:|------:|--------:|---------:|-----------------:|--------------:|
| step           | 6.36262e+06 |    243.397       |    142.332       |     1 |   156   |    239   |    335           | 743           |
| amount         | 6.36262e+06 | 179862           | 603858           |     0 | 13389.6 |  74871.9 | 208721           |   9.24455e+07 |
| oldbalanceOrg  | 6.36262e+06 | 833883           |      2.88824e+06 |     0 |     0   |  14208   | 107315           |   5.9585e+07  |
| newbalanceOrig | 6.36262e+06 | 855114           |      2.92405e+06 |     0 |     0   |      0   | 144258           |   4.9585e+07  |
| oldbalanceDest | 6.36262e+06 |      1.1007e+06  |      3.39918e+06 |     0 |     0   | 132706   | 943037           |   3.56016e+08 |
| newbalanceDest | 6.36262e+06 |      1.225e+06   |      3.67413e+06 |     0 |     0   | 214661   |      1.11191e+06 |   3.56179e+08 |
| isFraud        | 6.36262e+06 |      0.00129082  |      0.0359048   |     0 |     0   |      0   |      0           |   1           |
| isFlaggedFraud | 6.36262e+06 |      2.51469e-06 |      0.00158577  |     0 |     0   |      0   |      0           |   1           |

## 5. Summary Statistics (Categorical)
|          |   count |   unique | top         |    freq |
|:---------|--------:|---------:|:------------|--------:|
| type     | 6362620 |        5 | CASH_OUT    | 2237500 |
| nameOrig | 6362620 |  6353307 | C2098525306 |       3 |
| nameDest | 6362620 |  2722362 | C1286084959 |     113 |