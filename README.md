# econometrics-pack

Primary objective: Test out Tensorflow functionality by building a bunch of econometrics-related primitives.
- Linear Least Squares Estimator: implemented using only the matrix primitives in Tensorflow. 
  - Contains both homoskedastic and robust std. error estimations (using Huber-White errors) for coefficients.
  - Has hypothesis testing for regression coefficients (2 sided t-test).
