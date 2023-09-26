# Core-sets for Fair and Diverse Data Summarization
This is the code repository for the paper **Core-sets for Fair and Diverse Data Summarization**, to appear in NeurIPS 2023.

## Code structure and usage

* First, follow the procedure explained in the repository [1] and use the script `generate_dataset.py` to load the initial Reddit file with Reddit ids and time stamps, and using the Reddit API make a call to retrieve the text of these messages.
  * Download the **Reddit dataset** (raw_data.zip) from this paper and repository [1]. 
  * The messages text is extracted by using the [Reddit API](https://github.com/reddit-archive/reddit/wiki/OAuth2) and the user needs to request client id and secret id and provide it as arguments to this script.
  * The outcome is the input file augmented with the text of the actual message.

* `utils/` contains the functionalities related to diversity maximization, data preprocessing and loading the datasets once they are preprocessed.
  * `preprocessing.py` encodes the actual message into a metric space and add color class to the message.
  * `data_loader.py` loads the Reddit or MovieLens datasets already encoded in a metric space, select the appropriate columns and convert into `numpy` matrices that are used by the DM and FDM algorithms.
  * `diversity_maximization.py` contains the function for Diversity Maximization (DM), Core-set Construction and Fair Diversity Maximization (FDM): 
    - `dm_sum_pairwise`, `dm_min_pairwise`, and `dm_sum_nn` are Diversity Maximization (DM) approximation algorithms in the Sum-Pairwise, Min-Pairwise and Sum-NN notion of diversity, respectively.
    - `cs_construction_sum_pairwise`, `cs_construction_min_pairwise`, and `cs_construction_sum_nn` are the core-set construction algorithms in the Sum-Pairwise, Min-Pairwise and Sum-NN notion of diversity, respectively.
    - `fdm_sum_pairwise`, `fdm_min_pairwise`, and `fdm_sum_nn` are Fair Diversity Maximization (FDM) a.k.a <em>colored version</em> approximation algorithms in the Sum-Pairwise, Min-Pairwise and Sum-NN notion of diversity, respectively.

## References

[1] Henghui Zhu, Feng Nan, Zhiguo Wang, Ramesh Nallapati, and Bing Xiang. Who Did They Respond to? Conversation Structure Modeling Using Masked Hierarchical Transformer, In Proceedings of AAAI, 2020.
[code repository](https://github.com/henghuiz/MaskedHierarchicalTransformer)

## Citation

If you have used the paper, please cite as:

```
@inproceedings{coreset_fair_diverse_datasum_neurips2023,
  title={Core-sets for Fair and Diverse Data Summarization},
  author={Mahabadi, Sepideh and Trajanovski, Stojan},
  booktitle = {Neural Information Processing Systems (NeurIPS 2023)},
  volume = {37},
  year={2023}
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
