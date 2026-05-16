def main():
    # Import inside function to avoid circular imports during test collection
    from src.main.main import main as orchestrator_main

    orchestrator_main()


if __name__ == "__main__":
    main()
