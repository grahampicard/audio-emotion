from source.preprocessing_expanded_audio import match_expanded_dataset


if __name__ == "__main__":
    match_expanded_dataset(3, transforms=["stftdb", "chroma"])