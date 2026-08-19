"""Fixture lesson for the day-folder contract: prints one measured metric."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixture day lesson")
    parser.add_argument("--smoke", action="store_true", help="run in smoke mode")
    args = parser.parse_args()
    mode = "smoke" if args.smoke else "full"
    print(f"mode={mode} ttft_ms=12.5")


if __name__ == "__main__":
    main()