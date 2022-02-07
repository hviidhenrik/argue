from pathlib import Path

PATH_ROOT = Path(__file__).parents[2]
PATH_ARGUE = PATH_ROOT / "argue"
PATH_MODELS = PATH_ARGUE / "models"


def test_definitions():
    print(f"\nRoot path:                     {PATH_ROOT}")
    print(f"Src aka argue path:            {PATH_ARGUE}")
    print(f"Model files path:              {PATH_MODELS}")

    assert "sp-pdm-phd-ARGUE" in str(PATH_ROOT), "Root path not correct, check definitions."
    assert "sp-pdm-phd-ARGUE\\argue" in str(PATH_ARGUE), "Src aka argue path not correct, check definitions."
    assert "sp-pdm-phd-ARGUE\\argue\\models" in str(PATH_MODELS), "Model files path not correct, check definitions."

    print("\nDefinitions work!")


if __name__ == "__main__":
    test_definitions()
