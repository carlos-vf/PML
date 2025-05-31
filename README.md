# Probabilistic-Deep-Forest

> [!NOTE]  
> Use Python 3.7 (only versions 3.7, 3.8 and 3.9 are compatible with `deep-forest` package. However, I only manage to make it work for 3.7.)


## To Do

1. Set up PRF (prepare enviroment and test the code) (any dataset)
2. Set up DF (prepare enviroment and test the code) (any dataset)
3. Find a suitable dataset for PDF (we need data with uncertainty) (maybe the one from PRF is ok to start (?))
4. Create PDF (we must be carefull because DF uses two different types of RF. We must take this into account)
5. Data visualization and plot of the performace
6. Compare PDF with other models (PRF, DF, RF, NNs) and with more datasets (PRF's dataset is synthetic. We should search a real noisy dataset with known uncertainty)
7. OPTIONAL: Write a report. Not necesary but if the results are good (or even just acceptable / slightly worse than other models) I think we should write a brief paper-like report even if we dont deliver it
8. Pass PML with a 32



## Sources

- **Probabilistic Random Forest**:
  - Paper: https://arxiv.org/abs/1811.05994
  - Github: https://github.com/ireis/PRF
    
- **Deep Forest**:
  - Paper: https://arxiv.org/abs/1702.08835
  - Github: https://github.com/LAMDA-NJU/Deep-Forest
    
- **Datasets**:
  - Noisy dataset (from PRF authors): https://github.com/ireis/PRF/tree/master/PRF/examples/data
  - MACHO Project: https://wwwmacho.anu.edu.au/
