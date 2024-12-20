from sekvo.cli.main import cli, load_provider_commands


def main():
    load_provider_commands()
    cli()

if __name__ == "__main__":
    main()