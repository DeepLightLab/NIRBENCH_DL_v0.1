# External model drop-in folder

Copy a Keras model file into this directory and run it with
`benchmark_newmodel.py` from the repository root.

The builder must accept the number of spectral features as its first positional
argument and return an uncompiled `keras.Model`. The expected input shape is
`(spectral_features, 1)` and the output must contain one continuous value.
Use the function name `build_model`, or select another factory with
`--builder FUNCTION`.

When this folder contains one Python model file, run `python
benchmark_newmodel.py` from the repository root. Use `--model-file FILE.py`
to select among multiple files.

The benchmark owns data loading, training-only scaling, optimizer and loss,
five-fold epoch selection, seeded final fits, test scoring, and result output.
Model files placed here are executable Python code and should be reviewed before
running them.
