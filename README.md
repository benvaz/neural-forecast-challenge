# Neural Signal Forecasting — HDR SMood Challenge Y2

Multi-component Ridge ensemble predicting cortical neural activity for the [2025 HDR SMood Challenge](https://www.codabench.org/competitions/9854/).

## Structure

```
baselines/
  submissions/neural_forecast_blend/
    model.py              # inference (CodaBench submission)
    requirements.txt
  training/neural_forecast_blend/
    train_blend.py        # Ridge + Koopman weight training
    validate.py           # local validation
```

Model weights are not included due to size. Generate via `python baselines/training/neural_forecast_blend/train_blend.py`.

## Approach

Four Ridge regression variants blended per-timestep via learned weights:

- **NeighborRidge (NB)** — per-channel Ridge using K most-correlated neighbor channels
- **MultiBand (MB)** — Ridge on 6 summary statistics per channel per frequency band
- **RAW** — Ridge on raw temporal values across all bands
- **Koopman** — PCA-based latent dynamics correction on persistence forecast

Variance-based routing splits batches into day-specific (d1) and private (priv) paths with separate weight optimization.

## References

- Competition: [Neural Signal Forecasting Challenge](https://www.codabench.org/competitions/9854/)

## License

MIT
