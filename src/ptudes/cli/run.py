import click

from ptudes.cli.flyby import ptudes_flyby

@click.group(name="ptudes")
def ptudes_cli() -> None:
    """Ptudes (point tudes) commands"""
    pass

ptudes_cli.add_command(ptudes_flyby)

def main():
    ptudes_cli()

if __name__ == '__main__':
     main()
