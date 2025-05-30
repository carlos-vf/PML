# Probabilistic-Deep-Forest

## To Do

1. Set up PRF (prepare enviroment and test the code) (any dataset)
2. Set up DF (prepare enviroment and test the code) (any dataset)
3. Find a suitable dataset for PDF (we need data with uncertainty) (maybe the one from PRF is ok to start (?) - yes, looks like the safe bet. If it takes too long we'll add some constraint or reduce it)
4. Create PDF (we must be carefull because DF uses two different types of RF. We must take this into account)
5. Data visualization and plot of the performance
6. Compare PDF with other models (PRF, DF, RF, NNs) and with more datasets (PRF's dataset is synthetic. We should search a real noisy dataset with known uncertainty)
7. Make the presentation
8. OPTIONAL: Write a report. Not necesary but if the results are good (or even just acceptable / slightly worse than other models) I think we should write a brief paper-like report even if we dont deliver it
9. EVEN MORE OPTIONAL: Try with measurements assuming distribution != Gaussian (?) (should be "easily generalized to other distributions". Also finding a dataset for which it makes sense) 
10. Pass PML with a 32



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
 
- **Related work**: https://www.connectedpapers.com/main/7ea35b35392c6ef5738635cec7d17b24fe3e4f04/Deep-forest/graph
