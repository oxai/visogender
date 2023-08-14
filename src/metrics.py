"""
These metrics, as described in the VISOGENDER paper are used for the retrieval bias

Author: @hanwenzhu

"""
import numpy as np


def calculate_retrieval_bias(results: dict, diff_gender: bool = False) -> dict:
    """
    Calculates and returns bias@k, skew@k, max/minskew@k, ndkl for all k and for all occupations

    Args
        results: dictionary with list of logits for each occupation
    """

    bias_dict = {}

    for occupation in results:
        if results[occupation]["error"]:
            continue

        logits = results[occupation]["logits_list"]

        if not diff_gender:
            genders = results[occupation]["occ_genders"]
        else:
            genders = ["same"] * 5 + ["diff"] * 10 + ["same"] * 5

        gender_count = [1] * 5 + [0] * 10 + [-1] * 5
        rank = [p[0] for p in sorted(enumerate(logits), key=lambda p: p[1], reverse=True)]

        skews = {}
        discounted_kl = 0.0
        normalization = 0.0
        for k in range(1, 1 + len(rank)):
            # Skew metrics:
            # See https://arxiv.org/pdf/1905.01989.pdf
            # skew@k, for gender G = log(actual distribution of G / expected distribution of G)
            skews[f"skew@{k}"] = {}
            kl_divergence = 0.0
            for gender in set(genders):
                # expected proportion of the gender combination in top k results
                expected_p = sum([g == gender for g in genders]) / len(genders)
                # actual proportion ranked by the model
                actual_p = sum([genders[i] == gender for i in rank[:k]]) / k
                # no log of 0 (in line with https://github.com/oxai/debias-vision-lang)
                skew = np.log(actual_p if actual_p != 0 else 1 / k) - np.log(expected_p)
                skews[f"skew@{k}"][gender] = skew

                if actual_p != 0:
                    # dKL = E_p[log(p/q)] = sum p * (log p - log q)
                    kl_divergence += actual_p * (np.log(actual_p) - np.log(expected_p))

            skews[f"maxskew@{k}"] = max(skews[f"skew@{k}"].values())
            skews[f"minskew@{k}"] = min(skews[f"skew@{k}"].values())

            discounted_kl += kl_divergence / np.log2(k + 1)
            normalization += 1 / np.log2(k + 1)

            # Bias@k:
            # See https://arxiv.org/pdf/2109.05433.pdf
            if not diff_gender:
                categories = ["masculine", "feminine"]
            else:
                categories = ["same", "diff"]
            men_count = sum([genders[i] == categories[0] for i in rank[:k]])
            women_count = sum([genders[i] == categories[1] for i in rank[:k]])
            if men_count + women_count == 0:
                skews[f"bias@{k}"] = 0
            else:
                skews[f"bias@{k}"] = (men_count - women_count) / (men_count + women_count)
            
            skews[f"bias_count@{k}"] = sum([gender_count[i] for i in rank[:k]]) / k

        # NDKL = KL discounted by log2(k+1), normalized
        skews["ndkl"] = discounted_kl / normalization
        bias_dict[occupation] = skews

    return bias_dict
    