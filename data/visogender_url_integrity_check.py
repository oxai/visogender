import sys
import pandas as pd
import requests
from src.definitions import OP_data_filepath, OO_data_filepath

def test_url(url: str) -> str:
    """

    This function checks the validity of a given URL by making an HTTP request and inspecting the response.
    It determines the status of the URL based on the HTTP response status code, the presence of certain headers,
    and potential errors during the request process.

    Args:
        url (str): The URL from VISOGENDER to be tested.

    Returns:
        str: A status string indicating the result of the URL test.

    """

    if pd.isnull(url):
        return "Skipped"
    try:
        response = requests.get(url, headers={"User-Agent": "OxAIBot/1.0"})
        if response.status_code == 200:
            license_header = response.headers.get("License")
            if license_header and "creativecommons" in license_header.lower():
                return "Valid and downloadable with CC license"
            else:
                return "Valid and downloadable without CC license"
        else:
            return f"Returned status code {response.status_code}"
    except requests.exceptions.Timeout:
        return "Request timed out"
    except requests.exceptions.TooManyRedirects:
        return "Too many redirects"
    except requests.exceptions.RequestException as e:
        return f"Request error: {e}"
    except:
        return "Unknown error"


def check_visogender_url_integrity(visogender_df: pd.DataFrame):

    """
    This checks the integrity of URLs in a VISOGENDER DataFrame by doing the following:

    1. Adds a "URL Status" column to the DataFrame based on the result of the `test_url` function.
    2. Identifies invalid URLs and prints them.
    3. Detects duplicate URLs in the DataFrame and prints duplicate pairs.

    Args:
        visogender_df (pd.DataFrame): The input DataFrame containing Visogender data.

    Returns:
        None

    """

    visogender_df["URL Status"] = visogender_df[
        "URL type (Type NA if can't find)"
    ].apply(test_url)
    invalid_urls = visogender_df[
        (visogender_df["URL Status"] != "Valid and downloadable")
        & (visogender_df["URL Status"] != "Skipped")
    ]
    if not invalid_urls.empty:
        print("Invalid URLs found:")
        print(invalid_urls)

    # Find duplicate URLs in the dataframe
    duplicates = visogender_df[
        visogender_df.duplicated(["URL type (Type NA if can't find)"], keep=False)
    ]

    # Group duplicates by URL and print them out in pairs
    print("Checking duplicate URLs:")
    for url, group in duplicates.groupby("URL type (Type NA if can't find)"):
        if len(group) > 1:
            duplicate_idxs = ", ".join(group["IDX"].values)
            print(f"IDs {duplicate_idxs} are duplicates for URL: {url}\n")


if __name__ == "__main__":

    OO_tsv_path = OO_data_filepath
    OP_tsv_path = OP_data_filepath

    for tsv_file in [OO_tsv_path, OP_tsv_path]:
        print(f"Checking {tsv_file}")

        # Try to read the TSV file into a DataFrame, and handle any errors that occur
        try:
            visogender_dataframe = pd.read_csv(tsv_file, header=0, sep="\t")
        except FileNotFoundError:
            print(f"Error: file '{tsv_file}' not found.")
            sys.exit(1)
        except pd.errors.EmptyDataError:
            print(f"Error: file '{tsv_file}' is empty.")
            sys.exit(1)
        except pd.errors.ParserError as e:
            print(f"Error: could not parse file '{tsv_file}': {e}")
            sys.exit(1)

        # Call the test_urls function with the dataframe
        check_visogender_url_integrity(visogender_dataframe)
