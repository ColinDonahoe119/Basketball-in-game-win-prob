from pathlib import Path
import gdown

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Google Drive file IDs
RAW_FILES = {
    "teams.csv": "1y3h4m6txHKjATUCTiVJeR88LL8ntHaGH",
    "possessions.csv": "1yzK_U9cIJFBx3GOmSl6clNSJ6HcdHZfh",
    "teams_date_stats.csv": "1UrADG9cHl4ALU1rYjPRp22ty_VYcavSD",
}

PROCESSED_FILES = {
    "train.csv": "1jcz6VbRRMpqbXhOpU4kAD3j9YSjiyO-i",
    "test.csv": "1XKtBHMoexhY-SKMQoS7VjZtBjNFMXCoJ",
    "igwp_test.csv": "1YX9LWdAW2YvaqlI0-vhHPd-Fg_PcKX93",
    "game_comebacks_df.csv": "1xDVjoZyqEC2w9QegQVT1k3iaXqVaN_xG",
    "game_swings_df.csv": "1zrb5OJbAOrjqY7oVmC3fb_Kbn7Y-E8ts",
}


def download_files(files, destination):
    # Downloads files from Google Drive into the specified folder.
    # Existing files are skipped.

    destination.mkdir(parents=True, exist_ok=True)

    for filename, file_id in files.items():

        output_path = destination / filename

        if output_path.exists():
            print(f"✓ {filename} already exists.")
            continue

        print(f"Downloading {filename}...")

        url = f"https://drive.google.com/uc?id={file_id}"

        gdown.download(
            url,
            str(output_path),
            quiet=False
        )

    print("\nDownload complete.")


def download_raw_data():
    """Download the raw datasets."""
    download_files(RAW_FILES, RAW_DIR)


def download_processed_data():
    """Download the processed datasets."""
    download_files(PROCESSED_FILES, PROCESSED_DIR)


