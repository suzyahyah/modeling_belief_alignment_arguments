## Offer a Different Perspective: Modeling the Belief Alignment of Arguments in Multi-party Debates 

Code and data setup for the EMNLP 2022 paper: [pdf](https://aclanthology.org/2022.emnlp-main.818.pdf), [Poster](https://drive.google.com/file/d/1TXF03LnTtB2SkkJXooRurqbizbwlyoGR/view?usp=sharing)

### Getting Data
This paper uses the Change My View dataset and IQ2 Debates Corpus - see paper for descriptions about the datasets. We also need to download and prepare the embeddings

`bash bin/get_data.sh`

### Data Processing 

#### Processing CMV
`bash bin/prep_cmu_ids.sh`

#### Processing IQ2
`bash bin/debates-data-prep.sh`


### To Run Experiments:
`bash bin/run_exps.sh <dataset>`

where <dataset>={cmv,IQ2}

**Diclaimer:** Most of this code was first written in 2018-2019, and I don't recommend this way of setting up experimental parameters, organising code or logging results. The main method (VAE + triplet loss) which you might be looking for would be in `code/models/vae_model.py`.

### Citation 

If you found this paper helpful, please consider citing:

```
@inproceedings{sia2022offer,
  title={Offer a Different Perspective: Modeling the Belief Alignment of Arguments in Multi-party Debates},
  author={Sia, Suzanna and Jaidka, Kokil and Ahuja, Hansin and Chhaya, Niyati and Duh, Kevin},
  booktitle={Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing},
  pages={11939--11950},
  year={2022}
}
```
