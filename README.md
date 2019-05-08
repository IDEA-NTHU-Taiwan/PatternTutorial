# PatternTutorial
A Graph-based Pattern Representations with tutorial

#### Easy Reading: [nbviewer](https://nbviewer.jupyter.org/github/IDEA-NTHU-Taiwan/PatternTutorial/blob/master/Graph-based%20Pattern%20Representations%20Tutorial.ipynb)

#### Slides: [Google Slides](https://docs.google.com/presentation/d/1COyF_gAl3h3vl8RM-moZIfqDywsuZX1hFEBjWKiGsBQ)

#### GitHub Repo: [IDEA-NTHU-Taiwan/PatternTutorial](https://github.com/IDEA-NTHU-Taiwan/PatternTutorial)

#### Example Dataset: [SemEval 2017 Task](https://competitions.codalab.org/competitions/16380)

#### Author: [Ray](https://github.com/thisray), [Evan](https://github.com/EvanYu800112)

#### Developping
for developer
```
PatternTutorial$ pip3 install --user --editable .
```
#### Libraries requirements
`matplotlib`, `networkx`, `nltk`, `numpy`, `pandas`, `IDEAlib` (in this repo)

#### References

Argueta Carlos, Elvis Saravia, and Yi-Shin Chen. "Unsupervised graph-based patterns extraction for emotion classification." In 2015 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM), pp. 336-341. IEEE, 2015.

Argueta Carlos, Fernando H. Calderon, and Yi-Shin Chen. "Multilingual emotion classifier using unsupervised pattern extraction from microblog data." Intelligent Data Analysis 20, no. 6 (2016): 1477-1502.

Saravia Elvis, Carlos Argueta, and Yi-Shin Chen. "Unsupervised graph-based pattern extraction for multilingual emotion classification." Social Network Analysis and Mining 6, no. 1 (2016): 92.

Saravia Elvis, Hsien-Chi Toby Liu, Yen-Hao Huang, Junlin Wu, and Yi-Shin Chen. "CARER: Contextualized Affect Representations for Emotion Recognition." In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp. 3687-3697. 2018.



---

#### TODO

1. some column names are hard-code now (in `patternDict_()`)
2. graph minus
3. the multi-process lock problem in `patternDict()`
4. the hard-code args in multi-process
5. the rule of `both`
6. patternize function
7. make `token2cwsw()` prettier
8. Unit testing!!!!!! 

